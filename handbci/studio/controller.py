"""Device-independent lifecycle, timing, preview, marking and recording."""
from __future__ import annotations
from collections import deque
import copy
import json
from pathlib import Path
import queue
import threading
import time
import h5py
import numpy as np
from .devices import ADAPTERS
from .experiment import DEFAULT_PROTOCOL, Experiment, validate_protocol
from .storage import SessionWriter, paradigm_id
from .triggers import TriggerManager
from .vendor.dsp import DisplayFilter
from ..speech import SpeechCache


class DeviceClock:
    """Only EEG packets advance epoch; asynchronous events never advance it."""
    def __init__(self):
        self.last = None
        self.offset = 0

    def eeg(self, raw):
        reset = False
        if self.last is not None and raw < self.last:
            if self.last > 0xF0000000 and raw < 0x10000000:
                self.offset += 2**32
            else:
                reset = True
        self.last = raw
        return raw+self.offset, reset

    def event(self, raw):
        if self.last is None: return raw
        center = self.last+self.offset
        return min((raw+self.offset+k*2**32 for k in (-1,0,1)), key=lambda x: abs(x-center))


class Controller:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sessions_root = self.root/'runtime/studio/sessions'
        self.speech = SpeechCache(self.root/'runtime/audio_cache')
        self.lock, self.operation = threading.RLock(), threading.Lock()
        self.echo_cv = threading.Condition()
        self.echoes = deque(maxlen=256)
        self.packets, self.marks = queue.Queue(8192), queue.Queue(256)
        self.confirmations = queue.Queue(256)
        self.pending_marks = {}
        self.device, self.writer, self.experiment = None, None, None
        self.trigger_selected = False
        self.trigger = TriggerManager(self.local_event_row, self.wait_echo, lambda: self.device,
                                      self._packet, self.status_event)
        self.clock = DeviceClock()
        self.preview = deque(maxlen=4000)
        self.arrivals = deque()
        self.logs = deque(maxlen=100)
        self.impedances = []
        self.latest = None
        self.first_eeg_mono_ns = None
        self.stats = {'samples':0, 'packets':0, 'crc_error':0, 'tail_error':0, 'payload_mismatch':0,
                      'clock_gaps':0, 'clock_resets':0, 'sequence_jumps':0, 'input_highwater':0}
        self.last_summary, self.error, self.warning = {}, '', ''
        self.phase, self.phase_meta = 'idle', {}
        self.subject_state = {'revision':0,'text':'等待开始','subtitle':'','sound':''}
        self.audio = {}
        self.subject_lease, self.subject_heartbeat = False, 0
        self.protocol = copy.deepcopy(DEFAULT_PROTOCOL)
        self.consumer = threading.Thread(target=self._consume, name='studio-packet-consumer', daemon=True)
        self.marker = threading.Thread(target=self._mark_loop, name='studio-marker', daemon=True)
        self.confirmer = threading.Thread(target=self._confirm_loop, name='studio-mark-confirmation', daemon=True)
        self.consumer.start()
        self.marker.start()
        self.confirmer.start()

    def _packet(self, packet):
        try:
            self.packets.put_nowait(packet)
            self.stats['input_highwater'] = max(self.stats['input_highwater'], self.packets.qsize())
        except queue.Full:
            self.status_event('input_overflow', {'error':'接收队列已满，停止会话；原始断段保留。'})

    def status_event(self, kind, detail):
        with self.lock:
            self.stats[kind] = self.stats.get(kind, 0)+1
            marker_warning = detail.get('source') == 'triggerbox_acp' or kind == 'marker_warning'
            row = {'kind':'receive_fault', 'fault':kind, 'detail':detail, 'wall_time_ns':time.time_ns(),
                   'severity':'warning' if marker_warning else 'error'}
            if marker_warning:
                self.warning = '标记回传异常，脑电继续保存；异常标签将标为未确认。'
            self.logs.append(row)
            if self.writer:
                try: self.writer.submit('fault', row)
                except RuntimeError as exc: self.error = str(exc)
            if not marker_warning and kind in ('down','stalled','configuration_error','input_overflow','frame_error','clock_reset'):
                self.error = detail.get('error', detail.get('message', kind))
                if self.experiment: self.experiment.cancel(self.error)

    def _consume(self):
        while True:
            p = self.packets.get()
            try:
                with self.lock:
                    if p.raw_device_ms is None: p.raw_device_ms = p.device_ms
                    if p.kind == 'eeg':
                        p.device_ms, reset = self.clock.eeg(p.device_ms)
                        if reset: self.status_event('clock_reset', {'error':'设备毫秒计数重置，未人为拼接。'})
                        if self.latest:
                            prev = self.latest
                            expected = prev.device_ms+prev.counts.shape[1]*1000/self.device.settings['fs']
                            period = prev.counts.shape[1]*1000/self.device.settings['fs']
                            tol = .501 if period % 1 else .001
                            if abs(p.device_ms-expected) > tol:
                                self.stats['clock_gaps'] += 1
                                self.status_event('clock_gap', {'expected_ms':expected, 'actual_ms':p.device_ms})
                                self.display_filter.reset()
                            if p.packet_id != (prev.packet_id+1)&255: self.stats['sequence_jumps'] += 1
                        self.latest = p
                        n = p.counts.shape[1]
                        if self.first_eeg_mono_ns is None: self.first_eeg_mono_ns = p.mono_ns
                        self.stats['samples'] += n
                        self.stats['packets'] += 1
                        self.arrivals.append((p.mono_ns,n))
                        while self.arrivals and p.mono_ns-self.arrivals[0][0] > 3e9: self.arrivals.popleft()
                        display = self.display_filter.process(p.counts.astype(np.float32)*self.device.count_scale)
                        self.preview.append((p.device_ms, display))
                    else:
                        p.device_ms = self.clock.event(p.device_ms)
                    if p.kind == 'impedance': self.impedances = (p.detail or {}).get('kohm',[])
                    if self.writer: self.writer.submit('packet', p)
                    if p.kind == 'event':
                        row = {'kind':'device_event', 'event_id':p.code, 'device_time_ms':p.device_ms,
                            'raw_device_time_ms':p.raw_device_ms, 'wall_time_ns':p.wall_ns, 'mono_time_ns':p.mono_ns,
                            'recognized':p.code is not None, 'detail':p.detail, 'payload_hex':p.payload.hex()}
                        if self.writer: self.writer.submit('event',row)
                        self.logs.append(row)
                        if p.code is not None:
                            with self.echo_cv:
                                self.echoes.append(row)
                                self.echo_cv.notify_all()
            except BaseException as exc:
                self.status_event('frame_error', {'error':str(exc)})
            finally:
                self.packets.task_done()

    def wait_echo(self, code, after, timeout, source=None, before=None):
        end = time.perf_counter()+timeout
        with self.echo_cv:
            while True:
                upper = before() if callable(before) else before
                for row in self.echoes:
                    if (row['event_id'] == code and row['mono_time_ns'] >= after
                            and (upper is None or row['mono_time_ns'] < upper)
                            and (source is None or (row.get('detail') or {}).get('source') == source)):
                        return dict(row)
                left = end-time.perf_counter()
                if upper is not None: left = min(left, (upper-time.perf_counter_ns())/1e9)
                if left <= 0: return None
                self.echo_cv.wait(left)

    def idle_required(self):
        if self.writer is not None or self.experiment is not None: raise ValueError('正在记录或封存，设备和TriggerBox配置已锁定。')

    def connect(self, adapter, options):
        with self.operation:
            self.idle_required()
            if adapter not in ADAPTERS: raise ValueError('未知设备适配器。')
            self._disconnect()
            device = ADAPTERS[adapter](self._packet, self.status_event)
            with self.lock:
                self.device = device
                self.latest = None
                self.first_eeg_mono_ns = None
                self.clock = DeviceClock()
                self.preview.clear()
                self.arrivals.clear()
                self.impedances = []
                self.error = ''
                self.warning = ''
                self.stats = {k:0 for k in self.stats}
                self.display_filter = DisplayFilter(int(options.get('fs',1000)), int(options.get('channels',32)))
            device.connect(options)
            return device.describe()

    def _disconnect(self):
        self.trigger.disconnect()
        self.trigger_selected = False
        if self.device: self.device.disconnect()
        self.packets.join()

    def disconnect(self):
        with self.operation:
            self.idle_required()
            self._disconnect()

    def impedance(self, enabled):
        with self.operation:
            self.idle_required()
            if not self.device: raise ValueError('请先连接设备。')
            if self.trigger.wireless:
                self.trigger.disconnect()
                self.trigger_selected = False
            self.device.impedance(enabled)
            self.packets.join()
            with self.lock:
                self.latest = None
                self.preview.clear()
                self.clock = DeviceClock()
                self.display_filter = DisplayFilter(self.device.settings['fs'], self.device.settings['channels'])

    def trigger_connect(self, mode, port='', **options):
        with self.operation:
            self.idle_required()
            self.trigger_selected = False
            result = self.trigger.connect(mode, port, **options)
            self.trigger_selected = True
            return result

    def trigger_test(self):
        with self.operation:
            self.idle_required()
            error = ''
            try:
                return self.trigger.test()
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                row = {'kind':'trigger_check', 'wall_time_ns':time.time_ns(), 'error':error,
                       'trigger':self.trigger.state(), 'stats':dict(self.stats),
                       'synthetic':bool(self.device and self.device.synthetic),
                       'device_info':dict(self.device.info) if self.device else {}}
                with self.lock: self.logs.append(row)
                self._save_trigger_check(row)

    def _save_trigger_check(self, row):
        try:
            path = self.root/'runtime/studio/diagnostics/trigger_checks.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, ensure_ascii=False)+'\n')
        except OSError as exc:
            self.status_event('diagnostic_save_error', {'error':str(exc)})

    def prepare_audio(self, protocol):
        self.idle_required()
        p = validate_protocol(protocol)
        audio = self.speech.prepare(['开始','休息','结束']+[t['text'] for t in p['tasks']]) if p['sound'] else {'cues':[]}
        durations = {x['text']:x['duration_seconds'] for x in audio['cues']}
        if p['sound']:
            if durations['开始']+.4 > p['initial_rest_seconds'] or any(durations[t['text']]+.4 > p['prepare_seconds'] for t in p['tasks']):
                raise ValueError('开始语音须短于初始时长，任务语音须短于准备期，避免想象期仍有语音。')
            if durations['休息']+.4 > p['rest_seconds']: raise ValueError('休息时长须足够播放休息语音并预留0.4秒显示余量。')
        self.audio = {x['text']:x['url'] for x in audio['cues']}
        return audio

    def start(self, participant, protocol, require_subject=False):
        with self.operation:
            self.idle_required()
            p = validate_protocol(protocol)
            self.prepare_audio(p)
            if not self.device or not self.device.running or self.device.impedance_active:
                raise ValueError('请连接设备并结束阻抗模式。')
            if not self.trigger_selected: raise ValueError('请明确选择标记模式并连接。软件标记仅为估计。')
            if self.trigger.mode != 'software' and not self.trigger.state()['echo_verified']:
                raise ValueError('请先通过脑电流回传自检；串口握手或ACK不等于硬件对齐。')
            if not self.latest or time.perf_counter_ns()-self.latest.mono_ns > 1e9:
                raise ValueError('最近1秒没有脑电，不能开始记录。')
            if p['sound'] and any(t not in self.audio for t in ['开始','休息','结束']+[x['text'] for x in p['tasks']]):
                raise ValueError('请先准备并试听本机语音。')
            description = self.device.describe()
            cfg = json.loads((self.root/'configs/demo_6class.json').read_text(encoding='utf-8-sig'))
            cfg['device'] = {'source':'synthetic' if self.device.synthetic else 'neuracle_acp' if self.device.id == 'neusen_acp' else 'device_adapter',
                'adapter':self.device.id, 'sample_rate':self.device.settings['fs'], 'channels':self.device.labels,
                'stream_channels':len(self.device.labels)+1, 'unit':self.device.unit}
            cfg['classes'] = [{'id':i, 'name':t['text'], 'task_text':t['text'], 'action':t['action']} for i,t in enumerate(p['tasks'])]
            cfg['collection'], cfg['paradigm_id'] = dict(p), paradigm_id(p['tasks'])
            cfg['protocol_contract'] = {k:p[k] for k in ('prepare_seconds','task_seconds','rest_seconds','block_rest_seconds','initial_rest_seconds','randomize')}
            cfg['protocol_contract']['imagery_boundary'] = 'continuous_static_task_no_audio'
            exp = Experiment(self,p)
            p['sequence'] = exp.sequence
            cfg['session_metadata'] = {'participant_id':participant, 'triggerbox':self.trigger.state(),
                'recording_context': {'condition':p['condition'], 'notes':p['condition_notes']}}
            with self.lock:
                self.protocol, self.error, self.warning = p, '', ''
                self.pending_marks.clear()
                self.subject_lease, self.subject_heartbeat = require_subject, time.perf_counter()
                self.writer = SessionWriter(self.sessions_root, participant, description, p, cfg)
                self.experiment = exp
            exp.thread.start()
            return {'path':str(self.writer.path) if self.writer else str(self.last_summary.get('path',''))}

    def check_recording(self):
        if self.subject_lease and time.perf_counter()-self.subject_heartbeat > 2:
            raise RuntimeError('被试页面超过2秒未响应，保留记录并中止。')
        if self.error: raise RuntimeError(self.error)
        if self.writer and self.writer.error: raise RuntimeError(self.writer.error)
        if not self.latest or time.perf_counter_ns()-self.latest.mono_ns > 1e9:
            raise RuntimeError('1秒未收到脑电，保留文件并中止。')

    def set_phase(self, phase, row):
        with self.lock: self.phase, self.phase_meta = phase, dict(row)

    def subject(self, text, sound, subtitle=''):
        with self.lock:
            self.subject_state = {'revision':self.subject_state['revision']+1, 'text':text, 'subtitle':subtitle,
                'sound':self.audio.get(sound,'') if self.protocol['sound'] else '', 'host_mono_ns':time.perf_counter_ns()}

    def local_event_row(self, row):
        with self.lock:
            if self.writer: self.writer.submit('event',row)

    def local_event(self, kind, **data):
        now = time.perf_counter_ns()
        with self.lock:
            target = self.latest.device_ms+(now-self.latest.mono_ns)/1e6 if self.latest else None
            row = {'kind':kind,'device_time_ms':target,'wall_time_ns':time.time_ns(),'mono_time_ns':now, **data}
            if self.writer: self.writer.submit('event',row)
            return row

    def mark(self, kind, code, **data):
        row = {'kind':kind, 'event_id':code, 'wall_time_ns':time.time_ns(), 'mono_time_ns':time.perf_counter_ns(), **data}
        with self.lock:
            p = self.latest
            row['device_time_ms'] = p.device_ms+(row.get('planned_mono_ns',row['mono_time_ns'])-p.mono_ns)/1e6 if p else None
        try: self.marks.put_nowait(row)
        except queue.Full: raise RuntimeError('标记队列积压，停止会话。')

    def _mark_loop(self):
        while True:
            row = self.marks.get()
            try:
                row.update(acknowledged=False, echo_verified=False, marking='unverified_hardware_marker')
                row['dispatch_lag_ms'] = (time.perf_counter_ns()-row['mono_time_ns'])/1e6
                if self.trigger.mode != 'software' and row['dispatch_lag_ms'] > 100:
                    raise RuntimeError('硬件标记排队超过100毫秒，跳过过期发送并标为未确认；采集继续。')
                result = self.trigger.send(row['event_id'], wait_for_echo=False)
                row.update(result)
                if self.trigger.mode == 'software': row['marking'] = 'software_estimate'
                if row.get('verification_pending'):
                    with self.lock:
                        previous = self.pending_marks.get(row['event_id'])
                        if previous is not None: previous['echo_before_mono_ns'] = row['send_mono_ns']
                        self.pending_marks[row['event_id']] = row
                    with self.echo_cv: self.echo_cv.notify_all()
            except BaseException as exc:
                if self.trigger.last.get('send_mono_ns',0) >= row['mono_time_ns']:
                    row.update(self.trigger.last)
                row['error'] = str(exc)
                row['verification_pending'] = False
                row['echo_verified'] = False
                self.status_event('marker_warning', {'error':str(exc), 'event_id':row['event_id'],
                                                      'trial_id':row.get('trial_id')})
            finally:
                try:
                    self.confirmations.put_nowait(row)
                except queue.Full:
                    self.status_event('input_overflow', {'error':'标记确认队列已满，保留文件并中止。'})
                self.marks.task_done()

    def _confirm_loop(self):
        while True:
            row = self.confirmations.get()
            try:
                if row.get('verification_pending'):
                    try:
                        remaining = (row['send_mono_ns']+2_000_000_000-time.perf_counter_ns())/1e9
                        self.trigger.confirm(row, timeout=max(0., remaining),
                                             before=lambda: row.get('echo_before_mono_ns'))
                    except Exception as exc:
                        row.update(echo_verified=False, verification_pending=False,
                                   confirmation_error=str(exc))
                    if row.get('echo_verified'):
                        row['device_time_ms'] = row['observed_device_ms']
                        row['marking'] = row['verification']
                    else:
                        row['error'] = '未在2秒内确认有效硬件回传；保留原始记录，标签未确认，采集继续。'
                        self.status_event('marker_warning', {'error':row['error'], 'event_id':row['event_id'],
                                                              'trial_id':row.get('trial_id')})
                self.local_event_row(row)
            except BaseException as exc:
                self.status_event('frame_error', {'error':str(exc)})
            finally:
                with self.lock:
                    if self.pending_marks.get(row['event_id']) is row:
                        del self.pending_marks[row['event_id']]
                self.confirmations.task_done()

    def abort(self, message='操作员中止'):
        if self.experiment: self.experiment.cancel(message)

    def finish_recording(self, complete, error):
        self.marks.join()
        self.confirmations.join()
        with self.lock:
            writer, self.writer = self.writer, None
        try:
            if writer: self.last_summary = writer.close(complete and not self.error, error or self.error)
        except BaseException as exc:
            self.error = str(exc)
            self.last_summary = {'path':str(writer.path), 'complete':False, 'error':str(exc)}
        finally:
            with self.lock:
                self.experiment = None
                self.phase = 'finished' if self.last_summary.get('complete') else 'aborted'

    def state(self):
        with self.lock:
            now = time.perf_counter_ns()
            recent = [(t,n) for t,n in self.arrivals if now-t <= 3e9]
            span = min(3., (now-self.first_eeg_mono_ns)/1e9+self.device.settings['points']/self.device.settings['fs']) if recent and self.first_eeg_mono_ns is not None else 0
            rate = sum(n for _,n in recent)/span if span > .1 else 0
            return {'adapters':[{'id':c.id,'title':c.title,'capabilities':c.capabilities} for c in ADAPTERS.values()],
                'device':self.device.describe() if self.device else None, 'trigger':self.trigger.state(),
                'trigger_selected':self.trigger_selected, 'stats':{**self.stats,'points_per_second':round(rate,1),
                    'input_queue':self.packets.qsize(), 'disk_queue':self.writer.queue.qsize() if self.writer else 0,
                    'last_packet_id':self.latest.packet_id if self.latest else None,
                    'device_time_ms':self.latest.device_ms if self.latest else None,
                    'host_wall_ns':self.latest.wall_ns if self.latest else None,
                    'last_packet_age_ms':round((now-self.latest.mono_ns)/1e6,1) if self.latest else None},
                'recording':self.writer is not None or self.experiment is not None, 'phase':self.phase,
                'phase_meta':self.phase_meta, 'subject':dict(self.subject_state), 'error':self.error, 'warning':self.warning,
                'impedance':list(self.impedances), 'logs':list(self.logs)[-15:], 'last_session':self.last_summary,
                'protocol':self.protocol, 'display_filter':{'highpass':1,'lowpass':40,'notch':50}}

    def waveform(self):
        with self.lock:
            packets = list(self.preview)
            device = self.device
        if not packets or not device: return {'times':[],'values':[],'labels':[]}
        fs = device.settings['fs']
        end = packets[-1][0]
        packets = [(t,y) for t,y in packets if t >= end-8000]
        y = np.concatenate([y for _,y in packets],axis=1)
        t = np.concatenate([t+np.arange(y.shape[1])*1000/fs for t,y in packets])
        step = max(1,len(t)//1500)
        return {'times':t[::step].tolist(),'values':y[:,::step].tolist(),'labels':list(device.labels), 'unit':device.unit}

    def sessions(self):
        result = []
        for path in sorted(self.sessions_root.glob('*/session.json'), reverse=True)[:200]:
            try:
                meta = json.loads(path.read_text(encoding='utf-8'))
                result.append({'id':path.parent.name,'participant':meta['participant_id'], 'complete':meta.get('complete',False),
                    'condition':meta.get('recording_context',{}).get('condition','未记录'),
                    'training_compatible':meta.get('quality',{}).get('training_compatible',False),'paradigm_id':meta.get('paradigm_id','')})
            except (OSError, ValueError, KeyError): continue
        return result

    def review(self, identity, trial=0, start=None, duration=None):
        if Path(identity).name != identity or '/' in identity or '\\' in identity: raise ValueError('无效会话名称。')
        path = self.sessions_root/identity
        if not path.is_dir(): raise ValueError('找不到记录。')
        if self.writer and path == self.writer.path: raise ValueError('请结束记录后查看。')
        meta = json.loads((path/'session.json').read_text(encoding='utf-8'))
        with h5py.File(path/'data.h5','r') as h:
            events = [json.loads(x) for x in h['events_json'].asstr()[:]] if 'events_json' in h else []
            fs, total = float(h.attrs['srate']), h['eeg'].shape[1]
            begin, end = 0, total
            if trial:
                relevant = [e for e in events if e.get('trial_id') == int(trial) and e.get('sample_index') is not None]
                if not relevant: raise ValueError('该试次没有可映射标签。')
                begin, end = max(0,min(e['sample_index'] for e in relevant)-int(fs)), min(total,max(e['sample_index'] for e in relevant)+int(fs))
            if start is not None:
                begin = max(begin,int(float(start)*fs))
                end = min(end,begin+int(float(duration or 10)*fs))
            step = max(1,(end-begin)//2000)
            values = h['eeg'][:,begin:end:step]
            stamps = h['device_time_ms'][begin:end:step]
            # One-second reception bins are computed from all arrival timestamps, not decimated points.
            wall = h['arrival_mono_ns'][begin:end]
            bins, counts = np.unique(((wall-wall[0])//1000000000) if len(wall) else np.array([],dtype=int),return_counts=True)
        return {'metadata':meta,'events':events,'total_samples':total,'sample_start':begin,'step':step,
            'times':stamps.tolist(),'values':values.tolist(),'labels':meta['device']['labels'], 'unit':meta['device']['unit'],
            'reception':{'seconds':bins.tolist(),'points':counts.tolist()}}
