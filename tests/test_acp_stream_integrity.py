"""Only byte-identical redundant prefixes with a complete valid frame recover."""
import struct
import unittest

from handbci.studio.vendor import acp
from handbci.stream_recovery import StreamRecovery


def frame(sequence=1):
    body = bytes([2, 2, sequence])+struct.pack('<II', sequence*5, sequence*5)
    return acp.build_frame(acp.TOKEN_CMD, 17, acp.PKG_ALERT, body,
                           bytes((i*37)%256 for i in range(480)), acp.TAIL_CMD)


class StreamIntegrityTests(unittest.TestCase):
    def parse(self, chunks):
        seen, events = [], []
        link = acp.Link('', 0, seen.append, lambda k, d: events.append((k, d)))
        for chunk in chunks:
            link._received(chunk, 0)
        return seen, events

    def test_normal_frames_at_every_tcp_split(self):
        blob = frame()+frame(2)
        for split in range(1, len(blob)):
            seen, events = self.parse([blob[:split], blob[split:]])
            self.assertEqual([p.packet_id for p in seen], [1, 2])
            self.assertEqual(events, [])

    def test_exact_prefix_at_every_tcp_split_preserves_evidence_and_full_frame(self):
        original = frame()
        for prefix in (17, 114, 372, 443, 458, 486, 499):
            blob = original[:prefix]+original+frame(2)
            for split in range(1, len(blob)):
                seen, events = self.parse([blob[:split], blob[split:]])
                self.assertEqual([p.packet_id for p in seen], [1, 2], (prefix, split))
                self.assertEqual([k for k, _ in events], ['duplicate_prefix'], (prefix, split))
                detail = events[0][1]
                self.assertEqual(bytes.fromhex(detail['prefix_hex']), original[:prefix])
                self.assertEqual(bytes.fromhex(detail['verified_frame_hex']), original)
                self.assertEqual(detail['candidate_stream_offset'], 0)
                self.assertEqual(detail['repaired_samples'], 0)
                self.assertEqual(seen[0].payload, original[17:-4])

    def test_single_byte_reads(self):
        original = frame()
        seen, events = self.parse([bytes([b]) for b in original[:486]+original])
        self.assertEqual([p.packet_id for p in seen], [1])
        self.assertEqual([k for k, _ in events], ['duplicate_prefix'])

    def test_truncation_and_changed_prefix_remain_crc_faults(self):
        original = frame()
        altered = bytearray(original[:486]); altered[100] ^= 1
        for blob in (original[:372]+frame(2), bytes(altered)+original):
            seen, events = self.parse([blob])
            self.assertEqual(len(seen), 1)
            self.assertIn('crc_error', [k for k, _ in events])
            self.assertNotIn('duplicate_prefix', [k for k, _ in events])
            self.assertEqual(bytes.fromhex(events[0][1]['raw_receive_tail_hex']), blob)

    def test_invalid_following_crc_or_tail_is_never_accepted(self):
        original = frame()
        for index in (-4, -1, 490):
            broken = bytearray(original); broken[index] ^= 1
            seen, events = self.parse([original[:486]+broken+frame(2)])
            self.assertEqual([p.packet_id for p in seen], [2])
            self.assertIn('crc_error', [k for k, _ in events])
            self.assertNotIn('duplicate_prefix', [k for k, _ in events])

    def test_three_real_crc_faults_still_stop(self):
        _, events = self.parse([frame()[:372]+frame(2)])
        detail = next(d for k, d in events if k == 'crc_error')
        recovery = StreamRecovery(32, 5)
        recovery.request(detail, now=0)
        recovery.request(detail, now=1)
        with self.assertRaisesRegex(RuntimeError, '3次'):
            recovery.request(detail, now=2)


if __name__ == '__main__':
    unittest.main()
