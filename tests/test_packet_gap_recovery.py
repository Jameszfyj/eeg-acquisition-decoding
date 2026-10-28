"""Losses are recorded without stopping or inventing missing samples."""
import unittest
from types import SimpleNamespace

import numpy as np

from handbci.stream_recovery import ForwardGapRecovery, packet_discontinuity


def packet(ms, seq, points=5):
    return SimpleNamespace(device_ms=ms, packet_id=seq, counts=np.zeros((32, points)))


class PacketGapRecoveryTests(unittest.TestCase):
    def detail(self, missing_ms=10, seq=None):
        return packet_discontinuity(packet(1000, 254),
            packet(1005+missing_ms, (255+missing_ms//5) % 256 if seq is None else seq), 1000)

    def test_contiguous_packet_id_wrap_is_not_a_gap(self):
        self.assertIsNone(packet_discontinuity(packet(1000, 255), packet(1005, 0), 1000))

    def test_observed_five_ten_and_fifteen_ms_losses_can_recover(self):
        for missing_ms in (5, 10, 15):
            event = ForwardGapRecovery().accept(self.detail(missing_ms))
            self.assertEqual(event['missing_samples'], missing_ms)
            self.assertEqual(event['missing_frames'], missing_ms//5)
            self.assertEqual(event['repaired_samples'], 0)

    def test_regression_and_duplicate_still_stop(self):
        for ms, seq in ((1000, 254), (995, 253)):
            detail = packet_discontinuity(packet(1000, 254), packet(ms, seq), 1000)
            with self.subTest(ms=ms, seq=seq), self.assertRaises(RuntimeError):
                ForwardGapRecovery().accept(detail)

    def test_changed_packet_size_is_not_treated_as_loss(self):
        detail = packet_discontinuity(packet(1000, 254), packet(1015, 1, 10), 1000)
        with self.assertRaises(RuntimeError):
            ForwardGapRecovery().accept(detail)

    def test_repeated_and_large_losses_keep_valid_votes(self):
        recovery = ForwardGapRecovery()
        for i in range(100):
            event = recovery.accept(self.detail(5000 if i % 2 else 5), now=i/10)
            self.assertFalse(event['decision_votes_cleared'])
        self.assertEqual(event['recovery_number'], 100)
        self.assertEqual(event['missing_samples'], 5000)

    def test_uncorroborated_sequence_loss_is_flagged_not_repaired(self):
        for ms, seq in ((1005, 0), (1015, 255), (1016, 1)):
            detail = packet_discontinuity(packet(1000, 254), packet(ms, seq), 1000)
            event = ForwardGapRecovery().accept(detail)
            self.assertFalse(event['clock_id_consistent'])
            self.assertEqual(event['repaired_samples'], 0)


if __name__ == '__main__':
    unittest.main()
