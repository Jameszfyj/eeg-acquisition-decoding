"""Only small forward gaps corroborated by both device clocks and IDs recover."""
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

    def test_regression_duplicate_and_unmatched_clock_or_id_still_stop(self):
        for ms, seq in ((1000, 254), (995, 253), (1005, 0), (1015, 255), (1016, 1), (1110, 20)):
            detail = packet_discontinuity(packet(1000, 254), packet(ms, seq), 1000)
            with self.subTest(ms=ms, seq=seq), self.assertRaises(RuntimeError):
                ForwardGapRecovery().accept(detail)

    def test_changed_packet_size_is_not_treated_as_loss(self):
        detail = packet_discontinuity(packet(1000, 254), packet(1015, 1, 10), 1000)
        with self.assertRaises(RuntimeError):
            ForwardGapRecovery().accept(detail)

    def test_repeated_small_losses_stop_but_old_events_expire(self):
        recovery = ForwardGapRecovery()
        for i in range(10):
            recovery.accept(self.detail(5), now=i)
        with self.assertRaisesRegex(RuntimeError, '60秒'):
            recovery.accept(self.detail(5), now=10)
        self.assertEqual(recovery.accept(self.detail(5), now=70)['recovery_number'], 11)

    def test_total_lost_time_has_an_independent_limit(self):
        recovery = ForwardGapRecovery()
        recovery.accept(self.detail(100), now=0)
        recovery.accept(self.detail(100), now=1)
        with self.assertRaisesRegex(RuntimeError, '累计'):
            recovery.accept(self.detail(5), now=2)


if __name__ == '__main__':
    unittest.main()
