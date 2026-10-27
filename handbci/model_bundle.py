"""Versioned ONNX bundles: explicit EEG input contracts, no Python/pickle plugins."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import uuid
import zipfile

import numpy as np

from .config import validate_config, validate_participant_id, write_json
from .preprocess import StreamPreprocessor, filter_bank

FORMAT = 'handbci.onnx.v1'
MAX_UPLOAD = 32 * 1024 * 1024
MAX_MODEL = 128 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def validate_manifest(value):
    if not isinstance(value, dict) or value.get('format') != FORMAT:
        raise ValueError('模型包格式须为 handbci.onnx.v1。')
    m = copy.deepcopy(value)
    owner = validate_participant_id(m.get('participant_id', ''))
    cfg = validate_config(m['config'])
    if not 2 <= len(cfg['classes']) <= 6:
        raise ValueError('实时界面支持2至6类模型。')
    if cfg['device']['source'] not in ('neuracle_acp', 'synthetic_tcp'):
        raise ValueError('ONNX模型包支持ACP真实脑电或明确标记的TCP模拟输入。')
    if cfg['device']['source'] == 'neuracle_acp' and cfg['device']['unit'] != 'uV':
        raise ValueError('ACP模型必须明确使用uV输入。')
    if len(cfg['device']['channels']) not in (8, 16, 32, 64):
        raise ValueError('设备完整通道表须为8/16/32/64道；模型子集另在input.channels指定。')
    metadata = cfg.setdefault('session_metadata', {})
    if metadata.get('participant_id', owner) != owner:
        raise ValueError('模型包与配置中的被试编号不一致。')
    metadata.update(participant_id=owner, external_model=True, presentation='external_model',
                    input_contract='manifest-defined channels, units, sampling and preprocessing')
    inputs, outputs = m['input'], m['output']
    if inputs.get('pipeline') not in ('raw', 'fir', 'ifnet_filterbank'):
        raise ValueError('input.pipeline须为raw、fir或ifnet_filterbank。')
    channels = inputs.setdefault('channels', list(cfg['device']['channels']))
    if (not isinstance(channels, list) or not channels or len(channels) != len(set(channels))
            or any(c not in cfg['device']['channels'] for c in channels)):
        raise ValueError('模型输入通道须为完整设备通道表中不重复的名称。')
    for entry in (inputs, outputs):
        if not isinstance(entry.get('name'), str) or not entry['name'].strip():
            raise ValueError('必须填写ONNX输入及输出节点名称。')
    if outputs.get('kind') not in ('logits', 'probabilities'):
        raise ValueError('output.kind须明确填写logits或probabilities。')
    norm = inputs.get('normalization')
    if norm is not None:
        mean, std = np.asarray(norm['mean'], float), np.asarray(norm['std'], float)
        if (mean.shape != (len(channels),) or std.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError('标准化参数须与模型输入通道一一对应，且std为正有限数。')
    if not isinstance(m.get('name'), str) or not 1 <= len(m['name'].strip()) <= 100:
        raise ValueError('模型名称须为1至100个字符。')
    m.update(participant_id=owner, config=cfg)
    return m


def verify_graph(data):
    import onnx
    model = onnx.load_model_from_string(data)
    # Visit tensors in nested graphs/attributes too. Never resolve external files.
    def walk(message):
        if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError('请导出单文件ONNX；模型包不接受外部权重或文件路径。')
        for descriptor, value in message.ListFields():
            if descriptor.message_type:
                if descriptor.is_repeated:
                    for item in value: walk(item)
                else: walk(value)
    walk(model)
    onnx.checker.check_model(model)


def fingerprint(folder):
    h = hashlib.sha256()
    for name in ('manifest.json', 'model.onnx'):
        h.update(name.encode())
        with (Path(folder)/name).open('rb') as f:
            for data in iter(lambda: f.read(1024*1024), b''): h.update(data)
    return h.hexdigest()


class RawStream:
    def __init__(self, config):
        self.fs = self.output_fs = float(config['device']['sample_rate'])
        self.n_channels = len(config['device']['channels'])
        self.delay_seconds = 0.
        self.reset()

    def reset(self, first_source_index=0): self.expected_index = int(first_source_index)

    def process(self, raw, indices):
        raw, indices = np.asarray(raw, np.float32), np.asarray(indices, np.int64)
        if raw.shape != (self.n_channels, len(indices)) or not np.isfinite(raw).all():
            raise ValueError('脑电输入形状不符，或存在NaN/Inf。')
        if len(indices):
            if indices[0] != self.expected_index or np.any(np.diff(indices) != 1):
                raise ValueError('脑电不连续，须重置模型窗口。')
            self.expected_index = int(indices[-1])+1
        return raw, indices

    def window_bounds(self, retained, received_end_index):
        return dict(source_start_index=int(retained[0]), source_end_index=int(retained[-1]),
                    received_end_index=int(received_end_index), sample_rate=self.fs,
                    input_seconds=len(retained)/self.fs, filter_history_seconds=0.)


class ONNXDecoder:
    def __init__(self, folder, manifest=None):
        import onnxruntime as ort
        self.folder = Path(folder)
        self.manifest = validate_manifest(manifest or json.loads((self.folder/'manifest.json').read_text(encoding='utf-8')))
        m = self.manifest
        cfg = m['config']
        self.stream = RawStream(cfg) if m['input']['pipeline'] == 'raw' else StreamPreprocessor(cfg)
        self.order = [cfg['device']['channels'].index(c) for c in m['input']['channels']]
        self.n_time = round(cfg['window']['seconds']*self.stream.output_fs)
        self.n_classes = len(cfg['classes'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.folder/'model.onnx'), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        expected = [1, len(self.order), self.n_time]
        if m['input']['pipeline'] == 'ifnet_filterbank': expected.insert(1, 2)
        ins, outs = self.session.get_inputs(), self.session.get_outputs()
        if (len(ins) != 1 or len(outs) != 1 or ins[0].name != m['input']['name']
                or outs[0].name != m['output']['name'] or ins[0].type != 'tensor(float)'
                or len(ins[0].shape) != len(expected)
                or any(isinstance(a, int) and a != b for a,b in zip(ins[0].shape, expected))):
            raise ValueError('ONNX节点、float32类型或输入维度与模型包声明不一致。')
        self.predict(np.zeros(expected, np.float32))  # Check actual output before registration.
        self.recover_stream(0, 'initial')

    def predict(self, values):
        m = self.manifest
        output = np.asarray(self.session.run([m['output']['name']], {m['input']['name']:values})[0])
        if output.shape != (1, self.n_classes) or not np.isfinite(output).all():
            raise ValueError('模型必须输出有限的[1,类别数]数组。')
        p = output[0].astype(np.float64)
        if m['output']['kind'] == 'logits':
            p = np.exp(p-p.max());p /= p.sum()
        elif np.any(p < 0) or not np.isclose(p.sum(), 1., rtol=1e-5, atol=1e-6):
            raise ValueError('声明为probabilities的输出必须非负且总和为1。')
        return p

    def recover_stream(self, first_source_index, reason):
        self.stream.reset(first_source_index)
        self.history = np.empty((len(self.order), 0), np.float32)
        self.indices = np.empty(0, np.int64)
        self.last_prediction, self.last_window = -float('inf'), None

    def process(self, raw, indices):
        processed, retained = self.stream.process(raw, indices)
        if not len(retained): return None
        self.history = np.concatenate((self.history, processed[self.order]), axis=1)[:, -self.n_time:]
        self.indices = np.concatenate((self.indices, retained))[-self.n_time:]
        now = float(indices[-1])/self.stream.fs
        interval = self.manifest['config']['window']['stride_seconds']
        if self.history.shape[1] < self.n_time or now-self.last_prediction < interval-1e-9: return None
        values = self.history[None].copy()
        norm = self.manifest['input'].get('normalization')
        if norm: values = (values-np.asarray(norm['mean'],np.float32)[None,:,None])/np.asarray(norm['std'],np.float32)[None,:,None]
        if self.manifest['input']['pipeline'] == 'ifnet_filterbank': values = filter_bank(values, self.stream.output_fs)
        result = self.predict(np.ascontiguousarray(values, dtype=np.float32))
        self.last_prediction = now
        self.last_window = self.stream.window_bounds(self.indices, indices[-1])
        return now, result


def import_bundle(data, participant, *, registry, import_root, name=None):
    from .subject_online import CATALOG_FORMAT, read_registry
    participant = validate_participant_id(participant)
    if not 0 < len(data) <= MAX_UPLOAD: raise ValueError('模型包须不超过32MB。')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        entries = z.infolist()
        if len(entries) != 2 or {e.filename for e in entries} != {'manifest.json','model.onnx'}:
            raise ValueError('ZIP只能包含根目录manifest.json和model.onnx，不接受目录、额外文件或重复项。')
        if any(e.file_size > (256*1024 if e.filename == 'manifest.json' else MAX_MODEL) for e in entries):
            raise ValueError('模型包解压后的文件过大。')
        manifest = validate_manifest(json.loads(z.read('manifest.json')))
        model = z.read('model.onnx')
    if manifest['participant_id'] != participant: raise ValueError('模型包内部被试编号与填写编号不一致。')
    verify_graph(model)
    root = Path(import_root)
    folder = root/(time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]+'_'+participant)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        write_json(folder/'manifest.json', manifest)
        (folder/'model.onnx').write_bytes(model)
        ONNXDecoder(folder, manifest)
        digest = fingerprint(folder)
        catalog = read_registry(registry)
        for current in catalog['models']:
            if current['participant_id'] == participant and current['sha256'] == digest:
                shutil.rmtree(folder);return current
        try: path = (folder/'manifest.json').resolve().relative_to(ROOT).as_posix()
        except ValueError: path = str((folder/'manifest.json').resolve())
        entry = dict(participant_id=participant, model_id=participant+'_onnx_'+digest[:12],
                     name=name or manifest['name'], path=path, sha256=digest, runtime='onnx',
                     source=manifest['config']['device']['source'], n_channels=len(manifest['input']['channels']),
                     n_classes=len(manifest['config']['classes']))
        catalog['models'].append(entry)
        target = Path(registry);target.parent.mkdir(parents=True,exist_ok=True)
        temp = target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
        write_json(temp,catalog);temp.replace(target)
        return entry
    except BaseException:
        shutil.rmtree(folder);raise


def prepare_entry(entry, folder):
    folder = Path(folder)
    if fingerprint(folder) != entry['sha256']: raise ValueError('导入后的模型或输入约定发生变化，请重新导入。')
    m = validate_manifest(json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    if m['participant_id'] != entry['participant_id']: raise ValueError('模型内部被试不匹配。')
    cfg = m['config']
    cfg['session_metadata'].update(online_model_id=entry['model_id'], online_model_sha256=entry['sha256'])
    selection = dict(participant_id=entry['participant_id'],model_id=entry['model_id'],name=entry['name'],
        model=str(folder),sha256=entry['sha256'],runtime='onnx',source=cfg['device']['source'],
        synthetic_model=cfg['device']['source']=='synthetic_tcp',n_channels=len(m['input']['channels']),
        n_classes=len(cfg['classes']),sample_rate=cfg['device']['sample_rate'],driver='mock',physical_action=False)
    return selection,cfg


def main():
    p = argparse.ArgumentParser(description='将单文件ONNX和明确的输入约定打包为可导入ZIP')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    validate_manifest(json.loads(args.manifest.read_text(encoding='utf-8')))
    verify_graph(args.model.read_bytes())
    with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
        z.write(args.manifest,'manifest.json');z.write(args.model,'model.onnx')
    print(args.output)


if __name__=='__main__':main()
