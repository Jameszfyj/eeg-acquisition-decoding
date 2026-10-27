"""Portable ONNX export of the complete calibrated two-IFNet predictor."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy import signal
import torch
from torch import nn

from .config import write_json
from .retrained_runtime import load_portable

DEPLOYMENT_FORMAT = 'handbci.retrained_six_onnx.v1'
CALIBRATION_METADATA_KEY = 'handbci.retrained_calibration'


def alignment_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()


def _frozen_network(network):
    result=copy.deepcopy(network).cpu().eval()
    for module in result.modules():
        if getattr(module,'do_weight_norm',False):
            with torch.no_grad():
                module.weight.copy_(torch.renorm(module.weight,p=2,dim=0,maxnorm=module.max_norm))
            module.do_weight_norm=False
    result.requires_grad_(False)
    return result


class CalibratedRouter(nn.Module):
    """Input is the last 700 FIR samples at 250Hz in the 30 model channels.

    A fixed-length filtfilt operation is linear, including its odd padding and
    initial conditions. Matrices evaluated on the 500 identity basis vectors
    preserve that operation without custom ONNX operators or future samples.
    Float32 casts remain at the same boundaries as the original SciPy path.
    """
    def __init__(self,predictor):
        super().__init__()
        if predictor.calibration is None:
            raise ValueError('导出前必须加载配套20轮校准参数。')
        self.baseline=_frozen_network(predictor.models['baseline'])
        self.expert=_frozen_network(predictor.models['expert'])
        identity=np.eye(500,dtype=np.float64)
        notch=signal.filtfilt(*signal.iirnotch(50.,30.,fs=250.),identity,axis=-1)
        self.register_buffer('notch_matrix',torch.from_numpy(notch.copy()))
        for index,band in enumerate(((4.,16.),(16.,40.),(8.,13.),(13.,30.))):
            sos=signal.butter(4,band,fs=250.,btype='bandpass',output='sos')
            matrix=signal.sosfiltfilt(sos,identity,axis=-1)
            self.register_buffer('band_matrix_'+str(index),torch.from_numpy(matrix.copy()))
        self.register_buffer('alignment',torch.from_numpy(np.asarray(predictor.alignment,np.float64).copy()))
        self.residual=predictor.state is not None
        if self.residual:
            for key in ('mean','scale','delta'):
                self.register_buffer('residual_'+key,torch.from_numpy(np.asarray(predictor.state[key],np.float64).copy()))

    def _bands(self,windows,start):
        return torch.stack([torch.matmul(windows.double(),getattr(self,'band_matrix_'+str(start+i))).float()
                            for i in range(2)],dim=1)

    def forward(self,eeg):
        windows=torch.stack((eeg[:,:,:500],eeg[:,:,100:600],eeg[:,:,200:700]),dim=1).reshape(3,30,500)
        windows=torch.matmul(windows.double(),self.notch_matrix).float()
        baseline=self.baseline(self._bands(windows,0)).reshape(1,3,6)
        baseline_p=torch.softmax(baseline,dim=-1).mean(dim=1)
        expert=self._bands(windows,2)
        expert=torch.matmul(self.alignment,expert.double()).float()
        features=self.expert.stem(expert.flatten(1,2)).flatten(1)
        logits=self.expert.classifier(features).reshape(1,3,6).double()
        if self.residual:
            normalized=(features.double()-self.residual_mean)/self.residual_scale
            correction=torch.matmul(normalized,self.residual_delta[:-1])+self.residual_delta[-1]
            correction=correction.reshape(1,3,1)
            logits=torch.cat((logits[:,:,:4],logits[:,:,4:5]-.5*correction,
                              logits[:,:,5:6]+.5*correction),dim=-1)
        pair=torch.softmax(logits,dim=-1).mean(dim=1)[:,4:6]
        pair=(pair/pair.sum(dim=-1,keepdim=True)).float()
        routed=torch.cat((torch.zeros_like(baseline_p[:,:4]),pair),dim=-1)
        gate=baseline_p.argmax(dim=-1,keepdim=True)>=4
        return torch.where(gate,routed,baseline_p)


def build_manifest(predictor,cfg,calibration_path):
    cfg=copy.deepcopy(cfg)
    cfg['description']='909次训练的双IFNet及已拟合20轮EA/脚舌校准，完整三窗路由ONNX。'
    cfg['device'].update(host=None,port=None)
    cfg['window'].update(seconds=2.8,stride_seconds=.1,task_offset_seconds=0.)
    actions=('index','ring','thumb','little','pinch','middle')
    for category,action in zip(cfg['classes'],actions):
        category['hand_action']=action
    cfg['session_metadata']=dict(participant_id='P01',physical_event_time_verified=False,
        independent_online_validation=False)
    calibration=dict(rounds=20,trials=120,method=predictor.calibration['method'],
        model_sha256=predictor.sha256,
        parameters_sha256=hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
        alignment_sha256=alignment_sha256(predictor.alignment),
        source_session_id=predictor.calibration.get('source_provenance',{}).get('source_session_id'))
    return dict(format='handbci.onnx.v1',participant_id='P01',
        name='P01 · 909次训练＋20轮校准 · 完整ONNX',config=cfg,
        input=dict(name='eeg',pipeline='fir',channels=predictor.payload['channel_names']),
        output=dict(name='probabilities',kind='probabilities'),
        deployment=dict(format=DEPLOYMENT_FORMAT,training_tasks=909,
            window_samples=700,window_offsets_samples=[0,100,200],subwindow_samples=500,
            filter_implementation='fixed_500_filtfilt_float64_with_original_float32_boundaries',
            calibration=calibration))


def export_bundle(model_path,calibration_path,output):
    import onnx
    from .model_bundle import validate_manifest,verify_graph
    predictor,cfg=load_portable(model_path)
    predictor.restore_calibration(calibration_path)
    if predictor.payload['training'].get('n_tasks')!=909:
        raise ValueError('此发布入口要求已核对的909次训练模型。')
    graph=CalibratedRouter(predictor).eval()
    manifest=build_manifest(predictor,cfg,calibration_path)
    validate_manifest(manifest)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'model.onnx'
    torch.onnx.export(graph,torch.zeros(1,30,700),str(target),input_names=['eeg'],
        output_names=['probabilities'],opset_version=17,dynamo=False,do_constant_folding=True)
    model=onnx.load(str(target))
    entry=model.metadata_props.add();entry.key=CALIBRATION_METADATA_KEY
    entry.value=json.dumps(manifest['deployment']['calibration'],sort_keys=True,separators=(',',':'))
    onnx.save(model,str(target));verify_graph(target.read_bytes())
    write_json(output/'manifest.json',manifest)
    archive=output/'P01_909训练_20轮校准_ONNX.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        bundle.write(output/'manifest.json','manifest.json');bundle.write(target,'model.onnx')
    return archive


def main():
    parser=argparse.ArgumentParser(description='导出完整909次训练及20轮校准ONNX模型包')
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    print(export_bundle(args.model,args.calibration,args.output))


if __name__=='__main__':main()
