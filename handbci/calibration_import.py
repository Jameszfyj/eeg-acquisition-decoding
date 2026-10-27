"""Import labelled acquisition sessions locally, without executing uploaded code."""
from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import threading
import time
import uuid
import zipfile

import h5py
import numpy as np

from .calibration_store import device_contract
from .config import write_json
from .data import trials_from_events
from .shared_calibration import RawCalibrationData, SharedCalibrationStore, digest, signatures


def read_acquisition(folder, cfg, protocol):
    folder=Path(folder)
    meta=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    if meta.get('format')!='handbci.raw.v1' or meta.get('complete') is not True or meta.get('error'):
        raise ValueError('请选择已完整结束的本项目采集会话，不能导入未完成的记录。')
    owner=cfg['session_metadata']['participant_id']
    if meta.get('participant_id')!=owner:
        raise ValueError(f'采集数据不属于当前被试{owner}，请先选择匹配的模型。')
    source=meta['config']['device']; target=cfg['device']; collection=meta['protocol']
    device=meta['device']
    synthetic=device.get('synthetic')
    if not isinstance(synthetic,bool) or synthetic!=(target['source'] in ('synthetic','synthetic_tcp')):
        raise ValueError('人工与真实脑电不能混用。')
    if source.get('source')!=meta.get('source') or (not synthetic and meta['source']!='neuracle_acp'):
        raise ValueError('采集来源不一致或不是已支持的ACP来源。')
    if (source.get('unit')!=target['unit'] or source.get('sample_rate')!=target['sample_rate']
            or device.get('unit')!=target['unit'] or device['settings'].get('fs')!=target['sample_rate']
            or source.get('reference')!=target.get('reference')):
        raise ValueError('数据单位、采样率或参考设置与所选模型不一致。')
    channels=source['channels']
    if (len(set(channels))!=len(channels) or device.get('labels')!=channels
            or any(c not in channels for c in target['channels'])):
        raise ValueError('原始物理通道与记录不一致，或缺少模型所需通道。')
    identity=(dict(synthetic=True,channels=channels,sample_rate=target['sample_rate'],unit=target['unit'])
              if synthetic else device_contract(device))
    tasks=[t['text'] for t in collection['tasks']]
    wanted=[x[1] for x in signatures(cfg)]
    if len(tasks)!=len(set(tasks)) or len(wanted)!=len(set(wanted)) or set(tasks)!=set(wanted):
        raise ValueError('采集任务与所选模型不一致；不能把双手、双脚等不同任务改名混用。')
    rounds=protocol['rounds']
    if collection.get('loops')!=rounds:
        raise ValueError(f'当前要求每类{rounds}次，请导入{rounds}轮完整采集记录。')
    imagery=collection.get('task_seconds')
    prepare=collection.get('prepare_seconds')
    rest=collection.get('rest_seconds')
    if (any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in (imagery,prepare,rest))
            or not 4<=imagery<=30 or not 0<=prepare<=30 or not 0<=rest<=30):
        raise ValueError('采集时长无效；当前校准支持至少4秒的完整想象阶段。')
    quality=meta.get('quality',{})
    blocking=set(quality.get('reasons',[]))-{'device_event_time_semantics_unverified'}
    if blocking or quality.get('device_time_gaps') or quality.get('unverified_trial_ids'):
        raise ValueError('采集质量不满足校准：'+','.join(sorted(blocking))+'；请查看原始采集报告。')
    if any(f.get('severity','error')!='warning' for f in quality.get('receive_faults',[])):
        raise ValueError('原采集记录包含接收故障，不能登记为完整校准。')
    with h5py.File(folder/'data.h5','r') as h:
        if (h.attrs.get('handbci_format')!='handbci.raw.v1' or not h.attrs.get('complete')
                or json.loads(h.attrs['metadata_json'])!=meta):
            raise ValueError('session.json与data.h5的完整元数据不一致。')
        for name in ('eeg','source_indices','device_time_ms','events_json'):
            if name not in h or not isinstance(h.get(name,getlink=True),h5py.HardLink):
                raise ValueError('数据文件缺少原始字段，或包含不支持的外部链接。')
            ds=h[name]
            if not isinstance(ds,h5py.Dataset) or ds.is_virtual or ds.external:
                raise ValueError('不接收虚拟数据集或外部文件。')
        eeg=h['eeg']; index=h['source_indices']; stamps=h['device_time_ms']; events_ds=h['events_json']
        if (eeg.ndim!=2 or eeg.shape[0]!=len(channels) or eeg.dtype.kind not in 'fi'
                or index.shape!=(eeg.shape[1],) or index.dtype.kind not in 'iu'
                or stamps.shape!=index.shape or stamps.dtype.kind not in 'fi'
                or events_ds.ndim!=1 or events_ds.shape[0]>100000 or not eeg.shape[1]
                or eeg.size*max(4,eeg.dtype.itemsize)>2*1024**3):
            raise ValueError('数据维度、类型或大小不符合采集格式。')
        indices=index[:]; times=stamps[:]; fs=float(target['sample_rate'])
        pts=device['settings'].get('points',1)
        tolerance=.501 if (1000*pts/fs)%1 else .001
        if (not np.isfinite(times).all() or np.any(np.diff(indices)!=1)
                or np.any(np.abs(np.diff(times)-1000/fs)>tolerance)):
            raise ValueError('采集数据存在时间或采样索引缺口，不能用于完整校准。')
        events=[json.loads(e) for e in events_ds.asstr()[:]]
        boundaries=[e for e in events if e.get('kind') in ('cue','task_end')]
        if (len(boundaries)!=rounds*len(tasks)*2
                or any(type(e.get('sample_index')) is not int or e.get('echo_verified') is not True for e in boundaries)):
            raise ValueError('任务起止事件缺失或回传未核验，请保留有标签和回传完整的采集记录。')
        active=None
        for event in boundaries:
            if event['kind']=='cue':active=event
            elif active is None or event.get('label')!=active.get('label'):
                raise ValueError('任务结束标签与开始标签不匹配。')
            else:active=None
        trials=trials_from_events(boundaries,len(tasks))
        labels=np.asarray([t['label'] for t in trials])
        if (len(trials)!=rounds*len(tasks) or np.bincount(labels,minlength=len(tasks)).tolist()!=[rounds]*len(tasks)):
            raise ValueError('每类完整任务数不符合校准轮数。')
        for begin in range(0,len(trials),len(tasks)):
            if set(labels[begin:begin+len(tasks)])!=set(range(len(tasks))):
                raise ValueError('每轮必须包含每个任务各一次。')
        count=round(imagery*fs)
        for trial in trials:
            duration=trial['end']-trial['start']
            if abs(duration-imagery*fs)>max(.05*fs,2):
                raise ValueError('实际任务边界与想象时长不一致，请检查触发记录。')
            count=min(count,duration)
        # Retain only samples inside every recorded imagery boundary; never pad across rest.
        if count/fs<3.9:
            raise ValueError('想象窗口过短，无法保持原六分类模型所需的完整滤波支持。')
        imported=dict(protocol,imagery_seconds=float(imagery),rest_seconds=float(rest),
                      prepare_seconds=float(prepare),raw_samples_per_trial=int(count),external_acquisition=True)
        canonical=copy.deepcopy(cfg)
        data=RawCalibrationData(canonical,imported)
        order=[channels.index(c) for c in target['channels']]
        for trial in trials:
            first=int(np.searchsorted(indices,trial['start']))
            if (first+count>len(indices) or indices[first]!=trial['start']
                    or indices[first+count-1]>=trial['end']):
                raise ValueError('任务时间超出已保存的脑电，不能补点或拼接。')
            values=np.asarray(eeg[:,first:first+count],np.float32)[order]
            if not np.isfinite(values).all():raise ValueError('想象阶段脑电含NaN或无穷值。')
            data.windows.append(values)
            data.labels.append(wanted.index(tasks[trial['label']]))
            data.starts.append(int(trial['start']))
    return dict(data=data,protocol=imported,device=device,identity=identity,quality=quality,
                source_meta=meta,events=boundaries)


class CalibrationImports:
    MAX_CHUNK=8*1024**2
    MAX_DATA=2*1024**3
    MAX_META=4*1024**2

    def __init__(self, controller):
        self.controller=controller
        self.root=controller.output_root/'calibration_imports'
        self.lock=threading.RLock()
        self.sessions={}

    def _idle(self):
        if self.controller.state.get('running') or (self.controller.worker and self.controller.worker.is_alive()):
            raise ValueError('请先停止并保存当前会话，再导入校准数据。')
        if getattr(self.controller,'eeg_probe_busy',False) or getattr(self.controller,'eeg_discovery_busy',False):
            raise ValueError('请等待当前设备检验或搜索结束，再导入校准数据。')

    def _target(self, participant, model_id):
        from .six_class_online import SIX_MODEL_IDS, prepare as prepare_six
        from .subject_online import prepare_subject_model
        from .workbench_settings import calibration_protocol
        from .shared_calibration import imported_protocol
        if model_id in SIX_MODEL_IDS:
            selection,cfg=prepare_six(participant,model_id)
            if not selection.get('requires_calibration') or selection.get('right_hand_review'):
                raise ValueError('请选择有标签六组模型或已导入的ACP模型。')
            protocol=calibration_protocol(cfg)
        else:
            selection,cfg=prepare_subject_model(participant,model_id,registry=self.controller.registry)
            if selection.get('runtime')=='mi_studio_v2':
                raise ValueError('v2模型请在MI Studio v2中选择原始会话校准，完成后导入calibrated.pt。')
            if cfg['device']['source'] not in ('neuracle_acp','synthetic_tcp'):
                raise ValueError('当前模型不支持本项目采集数据校准。')
            protocol=imported_protocol(cfg)
        return selection,cfg,protocol

    def open(self, request):
        with self.lock,self.controller.condition:
            self._idle()
            self._target(request.get('participant'),request.get('model_id'))
            files=request.get('files')
            if not isinstance(files,list) or not 1<=len(files)<=2:raise ValueError('请选择一个ZIP或同一会话的两个数据文件。')
            names=[f.get('name') for f in files]
            if set(names) not in ({'session.json','data.h5'},{'session.zip'}) or len(set(names))!=len(names):
                raise ValueError('请选择包含session.json和data.h5的单个会话文件夹或ZIP。')
            for f in files:
                limit=self.MAX_META if f['name']=='session.json' else self.MAX_DATA
                if type(f.get('size')) is not int or not 0<f['size']<=limit:raise ValueError('数据文件大小无效，ZIP/H5最多2GB。')
            if len(self.sessions)>=4:raise ValueError('尚有未完成的导入，请取消后重试。')
            identifier=uuid.uuid4().hex
            folder=self.root/identifier;folder.mkdir(parents=True,exist_ok=False)
            for f in files:(folder/f['name']).touch(exist_ok=False)
            self.sessions[identifier]=dict(folder=folder,participant=request['participant'],model_id=request['model_id'],
                files={f['name']:dict(size=f['size'],received=0) for f in files},status='uploading')
            return dict(id=identifier,chunk_bytes=self.MAX_CHUNK)

    def _read(self, identifier):
        if identifier not in self.sessions:raise ValueError('导入已失效，请重新选择文件。')
        return self.sessions[identifier]

    def append(self, identifier, name, offset, data):
        with self.lock:
            state=self._read(identifier)
            if state['status']!='uploading' or name not in state['files']:raise ValueError('导入文件不匹配。')
            f=state['files'][name]
            if type(offset) is not int or offset!=f['received'] or not 0<len(data)<=self.MAX_CHUNK or offset+len(data)>f['size']:
                raise ValueError('上传分段不连续或文件大小不符，请重新导入。')
            with (state['folder']/name).open('ab') as handle:handle.write(data)
            f['received']+=len(data)
            return dict(received=f['received'])

    def _extract(self, folder):
        if not (folder/'session.zip').is_file():return
        with zipfile.ZipFile(folder/'session.zip') as archive:
            entries=archive.infolist()
            if len(entries)>2048:raise ValueError('ZIP文件条目过多，请只压缩一个采集会话。')
            selected={}
            for entry in entries:
                path=PurePosixPath(entry.filename)
                if ('\\' in entry.filename or path.is_absolute() or '..' in path.parts
                        or ':' in entry.filename or stat.S_ISLNK(entry.external_attr>>16)):
                    raise ValueError('ZIP含不安全的路径或链接。')
                if entry.is_dir():continue
                if path.name in ('session.json','data.h5'):
                    if path.name in selected:raise ValueError('ZIP包含多个会话，请每次只导入一份。')
                    limit=self.MAX_META if path.name=='session.json' else self.MAX_DATA
                    if not 0<entry.file_size<=limit:raise ValueError('ZIP解压后文件过大或为空。')
                    selected[path.name]=(entry,path.parent)
            if set(selected)!={'session.json','data.h5'} or selected['session.json'][1]!=selected['data.h5'][1]:
                raise ValueError('ZIP中须包含同一文件夹下的session.json和data.h5。')
            for name,(entry,_) in selected.items():
                with archive.open(entry) as source,(folder/name).open('xb') as target:
                    shutil.copyfileobj(source,target,self.MAX_CHUNK)

    def finish(self, request):
        with self.lock,self.controller.condition:
            self._idle();state=self._read(request.get('id'))
            if state['status']!='uploading' or any(f['received']!=f['size'] for f in state['files'].values()):
                raise ValueError('数据尚未全部上传。')
            state['status']='validating'
            identifier=request['id']; folder=state['folder']; final=None
            try:
                self._extract(folder)
                selection,cfg,protocol=self._target(state['participant'],state['model_id'])
                result=read_acquisition(folder,cfg,protocol)
                if selection.get('source')=='p01_six_acp_observation' or selection.get('runtime')=='retrained_six':
                    support=3.9
                else:
                    pipeline=(json.loads((Path(selection['model'])/'manifest.json').read_text(encoding='utf-8'))['input']['pipeline']
                              if selection.get('runtime')=='onnx' else 'fir')
                    history=0.
                    if pipeline!='raw':
                        prep=cfg['preprocess']
                        history=max((prep['numtaps']-1)/cfg['device']['sample_rate']+(prep['numtaps']-1)/prep['output_rate'],prep['trim_seconds'])
                    support=cfg['window']['seconds']+history
                if result['protocol']['raw_samples_per_trial']/cfg['device']['sample_rate']<support:
                    raise ValueError(f'这份想象数据不足以容纳模型窗口和滤波历史（需要至少{support:g}秒）。')
                hashes={name:digest(folder/name) for name in ('session.json','data.h5')}
                for old in self.controller.output_root.glob('*/calibration_import.json'):
                    report=json.loads(old.read_text(encoding='utf-8'))
                    if report.get('source_sha256')==hashes:
                        try:
                            summary=SharedCalibrationStore(self.controller.output_root).load(old.parent.name,cfg,protocol)['summary']
                            return dict(summary=summary,already_imported=True,message='这份校准数据已经导入，已选中原记录。')
                        except (ValueError,KeyError,OSError):pass
                output=folder/'converted';output.mkdir(exist_ok=False)
                write_json(output/'device.json',result['device'])
                write_json(output/'calibration.json',dict(complete=True,method='external_labelled_acquisition',
                    state=dict(ready=True,completed=protocol['trials'],total=protocol['trials']),
                    protocol=result['protocol'],events=result['events'],labels_used=True,
                    physical_event_time_verified=False,source_quality=result['quality']))
                write_json(output/'model_selection.json',dict(selection,config=cfg,external_calibration_data=True))
                result['data'].save(output)
                provenance=dict(format='handbci.calibration_import.v1',source_sha256=hashes,
                    source_session_id=result['source_meta'].get('session_id'),
                    original_prepare_seconds=result['protocol']['prepare_seconds'],
                    original_imagery_seconds=result['protocol']['imagery_seconds'],source_quality=result['quality'])
                write_json(output/'calibration_import.json',provenance)
                # Preserve original inputs for later verification, without touching the source files.
                original=output/'source';original.mkdir()
                for name in hashes:shutil.move(str(folder/name),str(original/name))
                sid=time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]+'_'+state['participant']
                destination=self.controller.output_root/sid
                output.rename(destination)
                final=destination
                summary=SharedCalibrationStore(self.controller.output_root).load(sid,cfg,protocol)['summary']
                return dict(summary=summary,already_imported=False,
                    message=f"导入完成：{protocol['rounds']}轮，准备{result['protocol']['prepare_seconds']:g}秒、想象{result['protocol']['imagery_seconds']:g}秒；未连接设备。")
            except zipfile.BadZipFile as error:
                raise ValueError('ZIP损坏或格式不正确，请重新压缩单个采集会话。') from error
            except Exception:
                if final is not None and final.is_dir():shutil.rmtree(final)
                raise
            finally:
                self.sessions.pop(identifier,None)
                shutil.rmtree(folder)

    def cancel(self, request):
        with self.lock:
            state=self.sessions.pop(request.get('id'),None)
            if state:shutil.rmtree(state['folder'])
            return dict(ok=True)
