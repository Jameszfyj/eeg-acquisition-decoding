"""Transport jitter must not change sample time, model inputs, or cue alignment."""
import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from examples.create_demo_model import build_manifest, build_model, bundle_bytes
from handbci.continuous_online import ContinuousController, VotingGate
from handbci.device_timeline import DeviceClockEstimate
from handbci.model_bundle import ONNXDecoder, import_bundle
from handbci.receive_timing import ReceiveTiming
from handbci.record_viewer import RecordStore, _lower_bound
from handbci.waveform_preview import WaveformPreview


class DeviceTimelineTests(unittest.TestCase):
    def test_delayed_packets_do_not_pull_software_cue_backwards(self):
        clock=DeviceClockEstimate(1000)
        clock.observe(5000,5,1_000_000_000)
        clock.observe(5005,5,1_705_000_000)  # 700 ms transport delay, only 5 ms on device.
        self.assertEqual(clock.estimate(1_720_000_000),5724)
        self.assertEqual(clock.snapshot(1_720_000_000)['transport_delay_ms'],700)
        clock.observe(5720,5,1_720_000_000)
        self.assertEqual(clock.estimate(1_720_000_000),5724)
        self.assertEqual(clock.snapshot(1_720_000_000)['transport_delay_ms'],0)

    def test_valid_delayed_slice_can_vote_in_the_current_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=ContinuousController(Path(tmp));gate=VotingGate(2)
            c._prediction(np.array([.8,.2]),1.,1.,gate,io.StringIO(),io.StringIO(),input_age_seconds=2.)
            self.assertEqual(gate.state['vote_total'],1)
            self.assertEqual(c.state['n_predictions'],1)

    def test_post_motion_cutoff_uses_device_clock_not_a_delayed_received_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=ContinuousController(Path(tmp));c.state['mode']='six'
            c.live_device_clock=DeviceClockEstimate(1000)
            c.live_device_clock.observe(5000,5,10_000_000_000)
            gate=VotingGate(2,cooldown_seconds=4)
            c._begin_fresh_window(12,gate)
            with patch('handbci.continuous_online.time.monotonic',return_value=16.1):
                self.assertFalse(c._fresh_window_ready(dict(received_end_index=3000,source_start_index=1000),gate))
                self.assertEqual(c.fresh_source_cutoff,6004)
                self.assertTrue(c._fresh_window_ready(dict(received_end_index=9000,source_start_index=6004),gate))

    def test_live_inference_has_identical_inputs_and_scores_despite_arrival_bursts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=build_manifest();manifest['input']['pipeline']='fir'
            entry=import_bundle(bundle_bytes(manifest,build_model(times=500)),
                'DEMO_ONNX',registry=root/'models.json',import_root=root/'models')
            raw=np.random.default_rng(42).normal(size=(8,7200)).astype(np.float32)*40
            outcomes=[]
            for delayed in (False,True):
                decoder=ONNXDecoder(Path(entry['path']).parent,prediction_stride_seconds=.5)
                controller=ContinuousController(root/'runs')
                timing=ReceiveTiming(1000);inputs=[];predictions=[];rates=deque()
                original=decoder.predict_window
                def capture(values):
                    inputs.append(values.copy());return original(values)
                decoder.predict_window=capture
                def prediction(p,*args,**kwargs):
                    predictions.append((dict(decoder.last_window),p.copy()))
                with patch.object(controller,'_prediction',side_effect=prediction):
                    for start in range(0,raw.shape[1],40):
                        end=start+40
                        # A pause followed by a burst, with unchanged samples and source indices.
                        now=10+end/1000
                        if delayed and 2200 <= end <= 3600:
                            now=max(now,13.6)
                        timing.received(40,round(now*1e9),100000+start)
                        with patch('handbci.continuous_online.time.monotonic',return_value=now):
                            controller._consume_chunk(raw[:,start:end],np.arange(start,end),10,decoder,
                                VotingGate(2),io.StringIO(),io.StringIO(),rates,timing,end)
                outcomes.append((inputs,predictions))
            self.assertGreater(len(outcomes[0][0]),5)
            self.assertEqual(len(outcomes[0][0]),len(outcomes[1][0]))
            for left,right in zip(outcomes[0][0],outcomes[1][0]):
                np.testing.assert_array_equal(left,right)
            for left,right in zip(outcomes[0][1],outcomes[1][1]):
                self.assertEqual(left[0],right[0])
                np.testing.assert_array_equal(left[1],right[1])

    def test_display_filter_keeps_device_timeline_and_retains_real_gap(self):
        raw=np.random.default_rng(4).normal(size=(3,2000))
        normal=WaveformPreview(['C3','Cz','C4'],1000)
        burst=WaveformPreview(['C3','Cz','C4'],1000)
        normal.append(raw,np.arange(2000))
        for start in range(0,2000,5):
            burst.append(raw[:,start:start+5],np.arange(start,start+5))
        np.testing.assert_allclose(normal.snapshot()['points'],burst.snapshot()['points'],atol=1e-12)
        burst.append(raw[:,:100],np.arange(2100,2200))
        points=np.asarray(burst.snapshot()['points'])
        self.assertLess(points[0,0],1.)
        self.assertAlmostEqual(np.diff(points[:,0]).max(),.105)

    def test_record_viewer_uses_device_clock_even_if_storage_indices_are_compacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'record';folder.mkdir()
            cfg=dict(device=dict(sample_rate=1000,channels=['C3'],unit='uV'),
                classes=[dict(id=0,task_text='test')])
            (folder/'session.json').write_text(json.dumps(dict(format='handbci.raw.v1',config=cfg,complete=True)))
            stamps=np.r_[np.arange(100),np.arange(200,300)]+1000.25
            with h5py.File(folder/'data.h5','w') as h:
                h.attrs.update(handbci_format='handbci.raw.v1',complete=True)
                h['eeg']=np.zeros((1,200));h['source_indices']=np.arange(200);h['device_time_ms']=stamps
                h.create_dataset('events_json',data=[json.dumps(dict(kind='cue',trial_id=1,label=0,sample_index=100)),
                    json.dumps(dict(kind='task_end',trial_id=1,label=0,sample_index=199))],dtype=h5py.string_dtype())
            store=RecordStore(root);info=store.details('record')
            self.assertAlmostEqual(info['duration'],.3)
            self.assertAlmostEqual(info['trials'][0]['start'],.2)
            data=store.window('record',[0],0,.3)
            self.assertEqual(data['gaps'],1)
            self.assertIn(None,data['traces'][0]['points'])
            self.assertEqual(store.window('record',[0],.2,.25)['n_samples'],50)
            self.assertEqual(_lower_bound(np.array([1.25,1.75,2.25]),1.5),1)


if __name__=='__main__':unittest.main()
