"""Live inference for the existing DY four-class IFNet checkpoints.

ACP samples are converted to uV by the studio adapter, then explicitly reordered
by electrode name before the original FIR/filter-bank processing. The historical
forwarding reference is not fully recorded; live equivalence is not certified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import load_config
from .preprocess import StreamPreprocessor, filter_bank

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'dy/质量分析_20261022/四分类训练'
CHANNELS = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6',
            'Cz','C3','C4','T7','T8','A1','A2','CP1','CP2','CP5','CP6','Pz',
            'P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']
NAMES = ['左握拳','左旋臂','右握拳','右旋臂']
SCALP = [i for i, ch in enumerate(CHANNELS) if ch not in ('A1','A2')]
CHECKPOINTS = {
    'dy-four-normal': ('DY 四类 · 普通条件', BASE/'分条件对照/单条件/models/dy_four_class_N_final.pt'),
    'dy-four-closed': ('DY 四类 · 闭眼条件', BASE/'分条件对照/单条件/models/dy_four_class_C_final.pt'),
    'dy-four-mixed': ('DY 四类 · 混合条件', BASE/'models/dy_four_class_final.pt'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def catalog():
    return [{'participant_id':'DY', 'model_id':key, 'name':name, 'path':str(path),
             'source':'dy_four_acp', 'n_classes':4, 'n_channels':30}
            for key,(name,path) in CHECKPOINTS.items() if path.is_file()]


def config():
    cfg = load_config(ROOT/'configs/demo_6class.json')
    cfg['description'] = 'DY已有四类模型的ACP实时观察；仅网页输出，无真实意图标签。'
    cfg['device'] = {'source':'neuracle_acp', 'host':'192.168.1.20', 'port':4097, 'sample_rate':1000.,
                     'stream_channels':33, 'channels':list(CHANNELS), 'unit':'uV'}
    cfg['classes'] = [{'id':i, 'name':name, 'task_text':name, 'action':'custom', 'action_name':name}
                      for i,name in enumerate(NAMES)]
    cfg['window'].update(seconds=2., stride_seconds=.1, task_offset_seconds=.75)
    cfg['session_metadata'] = {'participant_id':'DY', 'presentation':'continuous_four_class_monitor',
                               'input_contract':'ACP_uV_reordered_by_name_to_DY_CH32; original FIR and filter bank',
                               'historical_reference_equivalence_verified':False,
                               'unprompted':True}
    return cfg


def prepare(participant, model_id):
    if participant != 'DY' or model_id not in CHECKPOINTS:
        raise ValueError('请选择已有的DY四分类模型。')
    name, path = CHECKPOINTS[model_id]
    if not path.is_file(): raise ValueError('所选四分类模型文件不存在。')
    selection = {'participant_id':'DY', 'model_id':model_id, 'name':name, 'model':str(path),
                 'source':'dy_four_acp', 'synthetic_model':False, 'n_classes':4, 'n_channels':30,
                 'sample_rate':1000., 'sha256':digest(path), 'driver':'mock', 'physical_action':False}
    return selection, config()


def channel_order(actual):
    if len(actual) != 32 or len(set(actual)) != 32 or set(actual) != set(CHANNELS):
        raise ValueError('实际设备通道名称与四分类模型不匹配，未按编号猜测。')
    return [actual.index(name) for name in CHANNELS]


def load_checkpoint(path, expected_digest=None):
    import torch
    from .model import IFNetClassifier
    if expected_digest and digest(path) != expected_digest:
        raise ValueError('四分类权重在选择后发生变化，请重新选择并检查。')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'n_channels':30,'n_times':500,'n_classes':4,'radix':2,
                'out_planes':64,'kernel_size':63,'patch_size':125}
    prep = payload.get('preprocessing', {})
    if (payload.get('format') != 'dy.reference_ifnet_four_class.v1'
            or payload.get('architecture') != expected
            or payload.get('class_names') != NAMES
            or payload.get('channel_names') != [CHANNELS[i] for i in SCALP]
            or payload.get('sample_rate',250.) != 250.
            or payload.get('window_seconds') != 2.
            or payload.get('standardizer') is not None
            or prep.get('filter_bands') != [[4.,16.],[16.,40.]]
            or prep.get('raw_fs_declared') != 1000.
            or prep.get('output_fs_nominal') != 250.
            or prep.get('rereference') is not None):
        raise ValueError('四分类模型输入约定与本实时入口不一致。')
    model = IFNetClassifier(**expected).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    return payload, model


class FourClassDecoder:
    """Same continuous FIR and per-window filter bank as the trained model."""
    def __init__(self, payload, model, *, stream_config=None):
        self.stream = StreamPreprocessor(config() if stream_config is None else stream_config)
        self.model = model.eval()
        self.n_time = payload['architecture']['n_times']
        self.history = np.empty((30,0), np.float32)
        self.indices = np.empty(0, np.int64)
        self.last_window = None
        self.last_prediction = -float('inf')

    def recover_stream(self, first_source_index, reason):
        self.stream.reset(first_source_index)
        self.history = self.history[:, :0]
        self.indices = self.indices[:0]
        self.last_prediction = -float('inf')
        self.last_window = None

    def process(self, raw, indices):
        import torch
        processed, retained = self.stream.process(raw, indices)
        if not len(retained): return None
        self.history = np.concatenate((self.history, processed[SCALP]), axis=1)[:,-self.n_time:]
        self.indices = np.concatenate((self.indices, retained))[-self.n_time:]
        now = float(indices[-1])/self.stream.fs
        if self.history.shape[1] < self.n_time or now-self.last_prediction < .1-1e-9:
            return None
        self.last_window = self.stream.window_bounds(self.indices, indices[-1])
        inputs = self.window_inputs()
        with torch.inference_mode():
            probabilities = self.model(torch.from_numpy(inputs)).softmax(dim=1)[0].cpu().numpy()
        self.last_prediction = now
        return now, probabilities

    def window_inputs(self):
        return filter_bank(self.history[None], 250.)
