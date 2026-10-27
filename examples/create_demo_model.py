"""Create a tiny ONNX interface demo, not a trained EEG model."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build_manifest():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/demo_6class.json').read_text(encoding='utf-8-sig'))
    cfg['device'] = dict(source='synthetic_tcp',sample_rate=1000.,stream_channels=9,
        channels=['C3','Cz','C4','F3','F4','P3','Pz','P4'],unit='synthetic_uV',host='127.0.0.1',port=18712)
    cfg['classes'] = [dict(id=0,name='left',task_text='左手',action='custom'),
                      dict(id=1,name='right',task_text='右手',action='custom')]
    cfg['window'].update(seconds=2.,stride_seconds=.1)
    cfg['session_metadata'] = dict(participant_id='DEMO_ONNX')
    return dict(format='handbci.onnx.v1',name='ONNX接口演示（未训练，非实机模型）',participant_id='DEMO_ONNX',
                config=cfg,input=dict(name='eeg',pipeline='raw',channels=['C3','Cz','C4']),
                output=dict(name='scores',kind='logits'))


def build_model(channels=3, times=2000, n_classes=2):
    weights = np.zeros((channels,n_classes),np.float32)
    weights[0,0],weights[-1,-1] = .002,.002
    graph = helper.make_graph([
        helper.make_node('Mul',['eeg','eeg'],['squared']),
        helper.make_node('ReduceMean',['squared'],['power'],axes=[2],keepdims=0),
        helper.make_node('MatMul',['power','weights'],['scores'])],
        'interface_demo',[helper.make_tensor_value_info('eeg',TensorProto.FLOAT,[1,channels,times])],
        [helper.make_tensor_value_info('scores',TensorProto.FLOAT,[1,n_classes])],
        [numpy_helper.from_array(weights,'weights')])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def bundle_bytes(manifest=None, model=None):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(manifest or build_manifest(),ensure_ascii=False,indent=2))
        z.writestr('model.onnx',(model or build_model()).SerializeToString())
    return out.getvalue()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('runtime/demo_model.zip'))
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(bundle_bytes());print(args.output)
