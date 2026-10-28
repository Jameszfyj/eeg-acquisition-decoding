"""Packet loss must not reset a vote deadline or contribute artificial votes."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from handbci.continuous_online import ContinuousController, VotingGate, DEMO_CLASSES
from handbci.receive_timing import ReceiveTiming
from handbci.studio.controller import Controller


class LossVotingTests(unittest.TestCase):
    def test_deadline_uses_all_valid_slices_even_when_none_remain_in_rolling_window(self):
        gate = VotingGate(2)
        gate.update([.9,.1], 0.)
        gate.update([.8,.2], .1)
        self.assertIsNone(gate.tick(3.49))
        event = gate.tick(3.5)
        self.assertEqual(event['class_id'], 0)
        self.assertEqual(event['vote_counts'], [2,0])
        self.assertEqual(event['vote_total'], 2)
        self.assertEqual(event['decision_reason'], 'deadline_plurality')
        self.assertEqual(event['vote_scope'], 'all_valid_slices_in_round')
        np.testing.assert_allclose(event['mean_probabilities'], [.85,.15])
        self.assertIsNone(gate.tick(4.))  # No duplicate output or invented slice.

    def test_late_clean_slice_does_not_restart_deadline(self):
        gate = VotingGate(2)
        gate.update([.9,.1], 0.)
        gate.update([.8,.2], .1)
        gate.update([.1,.9], 3.)
        self.assertEqual(gate.round_started, 0.)
        self.assertEqual(gate.tick(3.5)['vote_counts'], [2,1])

    def test_empty_and_tied_rounds_expire_without_inventing_a_class(self):
        for tied in (False, True):
            gate = VotingGate(2)
            gate.start_round(0.)
            if tied:
                gate.update([.5,.5], 1.)
            self.assertIsNone(gate.tick(3.5))
            self.assertEqual(gate.state['decision_reason'], 'deadline_no_votes')
            self.assertIsNone(gate.round_started)
            gate.update([.8,.2], 4.)
            self.assertEqual(gate.round_started, 4.)

    def test_normal_majority_and_motion_inhibition_still_work(self):
        gate = VotingGate(2)
        events = [gate.update([.8,.2], t/10) for t in range(21)]
        self.assertEqual(sum(e is not None for e in events), 1)
        self.assertEqual(events[-1]['decision_reason'], 'majority')
        gate.update([.9,.1], 7.)
        gate.update([.9,.1], 7.1, inhibited=True)
        self.assertIsNone(gate.tick(20.))

    def test_controller_dispatches_saved_votes_without_a_new_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = ContinuousController(Path(tmp))
            controller.state.update(classes=DEMO_CLASSES[:2], mode='four', commands=[], probabilities=[])
            gate = VotingGate(2)
            gate.update([.2,.8], 0.)
            log = io.StringIO()
            controller._vote_tick(3.5, gate, log)
            command = json.loads(log.getvalue())
            self.assertEqual(command['class_id'], 1)
            self.assertEqual(command['vote_total'], 1)
            self.assertEqual(controller.state['n_predictions'], 0)
            controller._vote_tick(4., gate, log)
            self.assertEqual(len(log.getvalue().splitlines()), 1)

    def test_prompted_empty_round_finishes_trial_instead_of_reprompting(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = ContinuousController(Path(tmp))
            prompted = SimpleNamespace(phase='decoding', snapshot=lambda: {}, no_output=Mock())
            controller.prompted = prompted
            gate = VotingGate(2);gate.start_round(1.)
            controller._vote_tick(4.5, gate, io.StringIO())
            prompted.no_output.assert_called_once()

    def test_low_arrival_rate_from_losses_is_not_a_sample_rate_failure(self):
        timing = ReceiveTiming(1000)
        snapshot = dict(backlog_seconds=0, received_rate_hz=200, observation_seconds=3)
        for t in range(20):
            timing.check(snapshot, now=t, allow_packet_loss=True)
        timing.check(dict(snapshot, backlog_seconds=2), allow_packet_loss=True)
        with self.assertRaisesRegex(RuntimeError, '积压'):
            timing.check(dict(snapshot, backlog_seconds=2), allow_packet_loss=False)

    def test_offline_repeated_loss_and_stall_do_not_cancel_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = Controller(Path(tmp))
            controller.experiment = SimpleNamespace(cancel=Mock())
            controller.latest = SimpleNamespace(mono_ns=time.perf_counter_ns()-20_000_000_000)
            for _ in range(30):
                for kind in ('clock_gap','crc_error','tail_error','payload_mismatch','stalled'):
                    controller.status_event(kind, {})
                controller.check_recording()
            self.assertEqual(controller.error, '')
            controller.experiment.cancel.assert_not_called()
            controller.status_event('down', {'error':'peer closed'})
            controller.experiment.cancel.assert_called_once_with('peer closed')

    def test_generic_pt_decoder_recovery_reports_full_raw_support(self):
        from examples.create_demo_model import build_manifest
        from handbci.runtime import OnlineDecoder
        payload=dict(config=build_manifest()['config'], architecture=dict(n_channels=8,n_times=500))
        decoder=OnlineDecoder(payload,None)
        x=np.zeros((8,4000),np.float32)
        with patch('handbci.runtime.transform_windows', side_effect=lambda p,y:y), \
                patch('handbci.runtime.predict_inputs', return_value=np.array([[.8,.2]])):
            self.assertIsNotNone(decoder.process(x,np.arange(4000)))
            decoder.recover_stream(5005,'packet loss')
            self.assertIsNone(decoder.last_window)
            self.assertIsNone(decoder.process(x[:,:2000],np.arange(5005,7005)))
            self.assertIsNotNone(decoder.process(x[:,2000:],np.arange(7005,9005)))
            self.assertGreaterEqual(decoder.last_window['source_start_index'],5005)


if __name__ == '__main__':
    unittest.main()
