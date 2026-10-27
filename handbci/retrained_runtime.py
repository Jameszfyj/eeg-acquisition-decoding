"""Known two-IFNet PT runtime, using model-specific20-round calibration."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import uuid
import numpy as np
from scipy import signal

from .config import validate_config, write_json
from .four_class_online import CHANNELS, SCALP, digest
from .preprocess import StreamPreprocessor
from .six_class_online import NAMES
from .six_class_retrained import FORMAT, RetrainedSixClassModel

RUNTIME='retrained_six'
CALIBRATION_METHOD='twenty_round_EA_feet_tongue_residual_v1'


def load_portable(path,expected_sha256=None):
    model=RetrainedSixClassModel(path,expected_sha256)
    payload=model.payload
    if payload.get('participant_id')!='P01':raise ValueError('新六分类模型缺少正确的P01归属。')
    cfg=validate_config(payload['config'])
    if (cfg.get('session_metadata',{}).get('participant_id')!='P01'
            or cfg['device']['source']!='neuracle_acp' or cfg['device']['unit']!='uV'
            or cfg['device']['sample_rate']!=1000 or cfg['device']['channels']!=CHANNELS
            or [c['name'] for c in cfg['classes']]!=NAMES
            or [c.get('task_text') for c in cfg['classes']]!=NAMES
            or cfg['preprocess']!=dict(output_rate=250.,lowpass_hz=100.,highpass_hz=4.,numtaps=201,trim_seconds=1.)
            or cfg['window']['seconds']!=2. or cfg['window']['stride_seconds']!=.1
            or cfg['window']['task_offset_seconds']!=.6):
        raise ValueError('新模型的采样、预处理、通道或类别约定不匹配。')
    return model,cfg


def register_model(participant,path,registry,name=None,*,calibration_path=None):
    from .subject_online import ROOT, read_registry
    model,cfg=load_portable(path)
    if participant!='P01':raise ValueError('新六分类权重属于P01。')
    path=Path(path).resolve()
    try:stored=path.relative_to(ROOT).as_posix()
    except ValueError:stored=str(path)
    entry=dict(participant_id=participant,model_id='P01_retrained_'+model.sha256[:12],
        name=name or 'P01 · 909次重训＋20轮脚舌校准',path=stored,sha256=model.sha256,
        runtime=RUNTIME,source='neuracle_acp',n_channels=30,n_classes=6)
    if calibration_path is not None:
        calibration_path=Path(calibration_path).resolve()
        model.restore_calibration(calibration_path)
        calibration_sha=digest(calibration_path)
        try:calibration_stored=calibration_path.relative_to(ROOT).as_posix()
        except ValueError:calibration_stored=str(calibration_path)
        entry.update(model_id=entry['model_id']+'_cal_'+calibration_sha[:8],
            name=name or 'P01 · 909次重训 · 20轮校准已恢复',
            calibration_path=calibration_stored,calibration_sha256=calibration_sha)
    catalog=read_registry(registry)
    for current in catalog['models']:
        if current['model_id']==entry['model_id']:return current
    catalog['models'].append(entry)
    target=Path(registry);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    write_json(temporary,catalog);temporary.replace(target)
    return entry


def prepare_entry(entry,path):
    model,cfg=load_portable(path,entry['sha256'])
    if entry['participant_id']!='P01':raise ValueError('模型登记与内部被试不一致。')
    cfg=copy.deepcopy(cfg)
    cfg['session_metadata'].update(online_model_id=entry['model_id'],online_model_sha256=model.sha256)
    selection=dict(participant_id='P01',model_id=entry['model_id'],name=entry['name'],model=str(path),
        sha256=model.sha256,runtime=RUNTIME,source='neuracle_acp',synthetic_model=False,
        n_channels=30,n_classes=6,sample_rate=1000.,driver='mock',physical_action=False,
        requires_calibration=True,calibration_method=CALIBRATION_METHOD,
        offline_prototype_only=True,online_control_validated=False)
    if entry.get('calibration_path'):
        from .subject_online import resolve_path
        calibration_path=resolve_path(entry['calibration_path'])
        if digest(calibration_path)!=entry.get('calibration_sha256'):
            raise ValueError('配套20轮校准参数发生变化，请核对后重新登记。')
        model.restore_calibration(calibration_path)
        selection.update(requires_calibration=False,saved_calibration_ready=True,
            saved_calibration_path=str(calibration_path),saved_calibration_sha256=entry['calibration_sha256'],
            saved_calibration_method=model.calibration['method'],
            calibration_rounds=20,calibration_trials=120)
        # These are the published P01 gesture mappings in this model's original
        # output order. An explicit page selection can still override them.
        for c,action in zip(cfg['classes'],('index','ring','thumb','little','pinch','middle')):
            c['hand_action']=action
    return selection,cfg


def task_windows(raw,starts,cfg):
    """Reconstruct parent's .6/1/1.4s logical windows, inside imagery only."""
    raw=np.asarray(raw)
    if (raw.ndim!=3 or raw.shape[1]!=32 or len(starts)!=len(raw)
            or raw.shape[-1]<3900 or not np.isfinite(raw).all()):
        raise ValueError('专用校准需要完整的32路想象阶段脑电，至少3.9秒。')
    notch=signal.iirnotch(50.,30.,fs=250.);windows=[]
    for trial,start in zip(raw,starts):
        start=int(start);stream=StreamPreprocessor(cfg);stream.reset(start)
        values,retained=stream.process(trial,np.arange(start,start+trial.shape[1],dtype=np.int64))
        logical=(retained-start)/1000.-.5;parts=[]
        for offset in (.6,1.,1.4):
            j=int(np.searchsorted(logical,offset-1e-12))
            if j+500>len(retained) or retained[j]-1000<start or retained[j+499]>=start+trial.shape[1]:
                raise ValueError('校准想象阶段不足以容纳原三窗和完整滤波支持。')
            parts.append(values[SCALP,j:j+500])
        windows.append(signal.filtfilt(*notch,np.stack(parts),axis=-1).astype(np.float32))
    return np.stack(windows)


class RetrainedStreamDecoder:
    required_calibration_seconds=3.9

    def __init__(self,path,cfg,expected_sha256=None,*,calibration_path=None,calibration_sha256=None):
        self.predictor,_=load_portable(path,expected_sha256)
        if calibration_path is not None:
            if calibration_sha256 is None or digest(calibration_path)!=calibration_sha256:
                raise ValueError('配套20轮校准参数校验失败。')
            self.predictor.restore_calibration(calibration_path)
        self.cfg=cfg;self.stream=StreamPreprocessor(cfg)
        self.notch=signal.iirnotch(50.,30.,fs=250.)
        self.recover_stream(0,'initial')

    def recover_stream(self,first_source_index,reason):
        self.stream.reset(first_source_index)
        self.history=np.empty((30,0),np.float32);self.indices=np.empty(0,np.int64)
        self.last_prediction=-float('inf');self.last_window=self.last_details=None

    def fit_calibration(self,raw,labels,starts,source_summary=None):
        labels=np.asarray(labels,dtype=np.int64)
        windows=task_windows(raw,starts,self.cfg)
        summary=source_summary or {}
        provenance=dict(participant_id='P01',class_names=NAMES,
            channel_names=self.predictor.payload['channel_names'],
            source_sha256=summary.get('source_sha256') or {'raw_tasks':hashlib.sha256(np.asarray(raw).tobytes()).hexdigest()},
            shared_calibration_id=summary.get('id'),source_session_id=summary.get('source_session_id'),
            physical_event_time_verified=False)
        sid=provenance['source_session_id']
        if sid and any(sid in Path(p).parts for p in self.predictor.payload['training']['source_sha256']):
            raise ValueError('这组记录已经参与重训，不能作为新的独立校准记录。')
        self.predictor.calibrate(windows,labels,np.repeat(np.arange(1,21),6),provenance)

    def process(self,raw,indices):
        values,retained=self.stream.process(raw,indices)
        self.history=np.concatenate([self.history,values[SCALP]],axis=1)[:,-700:]
        self.indices=np.concatenate([self.indices,retained])[-700:]
        now=float(indices[-1])/self.stream.fs
        if len(self.indices)<700 or now-self.last_prediction<.1-1e-9:return None
        windows=np.stack([self.history[:,:500],self.history[:,100:600],self.history[:,200:700]])
        windows=signal.filtfilt(*self.notch,windows,axis=-1).astype(np.float32)
        output=self.predictor.predict(windows[None])
        self.last_window=self.stream.window_bounds(self.indices,indices[-1]);self.last_prediction=now
        self.last_details=dict(baseline_probabilities=output['baseline_probabilities'][0].tolist(),
            feet_tongue_probabilities=output['pair_probabilities'][0].tolist(),
            pair_routed=bool(output['feet_tongue_gate'][0]),
            calibration_method=CALIBRATION_METHOD,selected_head=self.predictor.calibration['method'])
        return now,output['probabilities'][0]

    def save_model_calibration(self,path):
        self.predictor.save_calibration(path)


def decoder_from_selection(selection,cfg):
    return RetrainedStreamDecoder(selection['model'],cfg,selection['sha256'],
        calibration_path=selection.get('saved_calibration_path'),
        calibration_sha256=selection.get('saved_calibration_sha256'))


def main():
    import argparse
    from .subject_online import DEFAULT_REGISTRY
    parser=argparse.ArgumentParser(description='登记909次重训模型及可选的已拟合20轮校准参数')
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--registry',type=Path,default=DEFAULT_REGISTRY)
    args=parser.parse_args()
    entry=register_model('P01',args.model,args.registry,calibration_path=args.calibration)
    print('已登记：'+entry['name']+'\n模型编号：'+entry['model_id'])


if __name__=='__main__':main()
