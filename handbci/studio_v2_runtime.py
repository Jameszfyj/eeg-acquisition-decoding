"""Explicit adapter for the optional, locally installed MI Studio v2 package."""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
from urllib.request import urlopen
import uuid

import numpy as np
from scipy.special import softmax

from .config import write_json
from .model_bundle import validate_manifest
from .preprocess import StreamPreprocessor
from .subject_online import ROOT, file_sha256, read_registry, resolve_path
from .workbench_settings import local_settings

RUNTIME='mi_studio_v2'
PROFILE='l100d250h4_pre_v2'


def tool_modules():
    options=local_settings().get('studio_v2',{})
    package=resolve_path(options.get('package_root','.'))
    if not (package/'mi_studio_v2/training.py').is_file():
        raise ValueError('本机尚未配置MI Studio v2校准工具。')
    if options.get('cargame_root'):
        os.environ['CARGAME_ROOT']=str(resolve_path(options['cargame_root']))
    os.environ['HANDBCI_ROOT']=str(ROOT)
    if str(package) not in sys.path:sys.path.insert(0,str(package))
    training=importlib.import_module('mi_studio_v2.training')
    contracts=importlib.import_module('mi_studio_v2.contracts')
    if Path(training.__file__).resolve().parent != (package/'mi_studio_v2').resolve():
        raise ValueError('载入的校准工具与本机配置路径不一致，请重启服务。')
    return training,contracts


def load_source(path):
    import torch
    training,contracts=tool_modules()
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,
            type(np.dtype(np.float64)),type(np.dtype(np.float32)),type(np.dtype(np.int64))]):
        payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('format')!=contracts.MODEL_FORMAT:
        raise ValueError('不是MI Studio六类源模型。')
    model=training.SourceModel(payload,device='cpu',source_path=path)
    profile,_=model.profile()
    if profile is None or profile['id']!=PROFILE:
        raise ValueError('此适配器只处理prepare锚点v2模型。')
    if (model.arch.get('protocol')!='IFNetRTG-L' or model.arch.get('n_class')!=6
            or model.arch.get('n_c')!=30 or model.arch.get('n_p')!=500 or model.arch.get('fs')!=250):
        raise ValueError('MI Studio v2网络输入结构不符。')
    return model,contracts


def register_model(participant,path,manifest_path,registry,name=None):
    if participant!='P01':raise ValueError('这份本机源模型的部署包属于P01。')
    path,manifest_path=Path(path).resolve(),Path(manifest_path).resolve()
    model,contracts=load_source(path)
    manifest=validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8-sig')))
    if (manifest['participant_id']!=participant or manifest.get('studio',{}).get('profile')!=PROFILE
            or manifest['input']['pipeline']!='ifnet_filterbank'
            or manifest['input']['channels']!=model.channels
            or [c['name'] for c in manifest['config']['classes']]!=model.class_names
            or manifest.get('model_labels')!=model.model_labels):
        raise ValueError('PT模型、部署配置的通道、类别或取窗规则不匹配。')
    sha=file_sha256(path)
    # Store the deploy contract together with the local checkpoint registration.
    target=ROOT/'runtime/models/studio_v2'/sha[:16]
    target.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(manifest['config'])
    cfg['session_metadata'].update(studio_profile=PROFILE,studio_calibration_complete=model.calibration is not None)
    if (target/'config.json').is_file() and json.loads((target/'config.json').read_text(encoding='utf-8'))!=cfg:
        raise ValueError('同一v2模型已经登记了不同输入配置，请先核对。')
    write_json(target/'config.json',cfg)
    config_sha=file_sha256(target/'config.json')
    entry=dict(participant_id=participant,model_id=participant+'_studio_v2_'+sha[:12],
        name=name or ('六类v2 · 专用校准完成' if model.calibration else '六类v2 · 待专用校准'),
        path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),sha256=sha,
        config_path=(target/'config.json').relative_to(ROOT).as_posix(),config_sha256=config_sha,
        runtime=RUNTIME,source='neuracle_acp',n_channels=30,n_classes=6,calibration_ready=model.calibration is not None)
    catalog=read_registry(registry)
    for previous in catalog['models']:
        if previous['model_id']==entry['model_id']:return previous
    catalog['models'].append(entry)
    target=Path(registry);temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    write_json(temporary,catalog);temporary.replace(target)
    return entry


def prepare_entry(entry,path):
    if file_sha256(path)!=entry['sha256']:raise ValueError('登记后的v2模型发生变化。')
    config_path=resolve_path(entry['config_path'])
    if file_sha256(config_path)!=entry['config_sha256']:raise ValueError('v2模型输入配置发生变化。')
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    cfg['session_metadata'].update(online_model_id=entry['model_id'],online_model_sha256=entry['sha256'])
    selection=dict(participant_id=entry['participant_id'],model_id=entry['model_id'],name=entry['name'],
        model=str(path),sha256=entry['sha256'],runtime=RUNTIME,source='neuracle_acp',synthetic_model=False,
        n_channels=30,n_classes=6,sample_rate=1000.,driver='mock',physical_action=False,
        studio_calibration_ready=entry['calibration_ready'],calibration_method='studio_frozen_latent_readout_v2',
        preprocessing_profile=PROFILE,external_calibration_only=True)
    return selection,cfg


class StudioV2Decoder:
    def __init__(self,path,cfg,expected_sha256=None):
        if expected_sha256 and file_sha256(path)!=expected_sha256:raise ValueError('v2模型校验失败。')
        self.model,self.contracts=load_source(path)
        self.calibration_disabled=bool(cfg.get('session_metadata',{}).get('online_calibration_disabled'))
        if self.model.calibration is None and not self.calibration_disabled:
            raise ValueError('v2源模型尚未专用校准；请在校准工具完成后导入calibrated.pt。')
        self.cfg=cfg;self.stream=StreamPreprocessor(cfg)
        self.prediction_stride_seconds=float(cfg['window']['stride_seconds'])
        if (not np.isfinite(self.prediction_stride_seconds) or self.prediction_stride_seconds <= 0
                or abs(self.prediction_stride_seconds*self.stream.output_fs-round(self.prediction_stride_seconds*self.stream.output_fs))>1e-6):
            raise ValueError('v2预测步长须为输出采样间隔的正整数倍。')
        self.prediction_step=round(self.prediction_stride_seconds*self.stream.fs)
        self.order=[cfg['device']['channels'].index(c) for c in self.model.channels]
        self.prompted=False
        self.recover_stream(0,'initial')

    def recover_stream(self,first_source_index,reason):
        self.stream.reset(first_source_index)
        self.history=np.empty((len(self.order),0),np.float32);self.indices=np.empty(0,np.int64)
        self.anchor=None;self.next_start=None;self.last_prediction=-float('inf')
        self.last_window=self.last_details=None

    def set_prompt_anchor(self,source_index):
        self.prompted=True
        snapped=round(source_index/4)*4 if source_index is not None else None
        if snapped!=self.anchor:
            self.anchor=snapped;self.next_start=snapped

    def process(self,raw,indices):
        values,retained=self.stream.process(raw,indices)
        self.history=np.concatenate([self.history,values[self.order]],axis=1)[:,-2000:]
        self.indices=np.concatenate([self.indices,retained])[-2000:]
        if len(self.indices)<500:return None
        if self.prompted:
            if self.anchor is None:return None
            starts=[]
            while self.next_start+1996<=self.indices[-1]:
                starts.append(self.next_start);self.next_start+=self.prediction_step
            if not starts:return None
            start=starts[-1]
            first=int(np.searchsorted(self.indices,start))
            if first+500>len(self.indices) or self.indices[first]!=start:return None
        else:
            if (indices[-1]/self.stream.fs)-self.last_prediction<self.prediction_stride_seconds-1e-9:return None
            first=len(self.indices)-500
        window=self.history[:,first:first+500]
        z=self.model.predict_logits(window[None])
        self.last_prediction=float(indices[-1])/self.stream.fs
        self.last_window=self.stream.window_bounds(self.indices[first:first+500],indices[-1])
        self.last_window.update(anchor='prepare' if self.prompted else 'continuous',
            model_start_index=int(self.indices[first]),prepare_source_index=self.anchor,
            profile=PROFILE)
        self.last_details=dict(calibration_method='studio_frozen_latent_readout_v2' if self.model.calibration is not None else None,
            online_calibration_disabled=self.calibration_disabled,
            selected_head='calibrated' if self.model.calibration is not None else 'original_source_head',
            preprocessing_profile=PROFILE,task_labels_used_for_live_prediction=False)
        return self.last_prediction,softmax(z[0])


def main():
    import argparse
    parser=argparse.ArgumentParser(description='打开本机MI Studio v2校准工具')
    parser.add_argument('--port',type=int,default=57500)
    args=parser.parse_args()
    tool_modules()
    package=resolve_path(local_settings()['studio_v2'].get('package_root','.'))
    os.environ['PYTHONPATH']=os.pathsep.join([str(package),str(ROOT),os.environ.get('PYTHONPATH','')])
    os.environ['HANDBCI_ROOT']=str(ROOT)
    os.environ['MI_STUDIO_ROOT']=str(ROOT)
    os.environ['MI_STUDIO_SOURCE_MODEL']=str(resolve_path(local_settings()['studio_v2']['source_model']))
    from mi_studio_v2.web.server import serve
    return serve(host='127.0.0.1',port=args.port,root=ROOT)


def ensure_tool_server():
    options=local_settings().get('studio_v2')
    if not options:return
    port=int(options.get('port',57500))
    if not 1<=port<=65535:raise ValueError('校准工具端口无效。')
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1.5) as response:
            info=json.load(response)
        if info.get('version')!='mi_studio/2.0' or Path(info.get('root','')).resolve()!=ROOT:
            raise ValueError('校准工具端口已被另一工作台占用。')
        return
    except OSError:
        pass
    folder=ROOT/'runtime/studio_v2';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'server.stdout.log').open('ab') as output,(folder/'server.stderr.log').open('ab') as errors:
        subprocess.Popen([sys.executable,'-B','-u','-X','utf8','-m','handbci.studio_v2_runtime','--port',str(port)],
            cwd=ROOT,stdout=output,stderr=errors,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


if __name__=='__main__':main()
