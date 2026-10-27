"""Inspire hand over the supplied AA AA ... 55 55 USB-to-CAN adapter.

Verified on COM3, 115200, hand ID 1. This is NOT a native RS485 driver.
Only pyserial is required. Importing/connecting never commands movement.
Six-channel order: little, ring, middle, index, thumb bend, thumb rotation.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import threading
import time
from typing import Callable, Sequence

import serial
from serial.tools import list_ports

CHANNELS = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
REGISTERS = {
    "targets": (1486, 12, "h"), "force_limits": (1498, 12, "h"),
    "speeds": (1522, 12, "h"), "angles": (1546, 12, "h"),
    "forces": (1582, 12, "h"), "currents": (1594, 12, "h"),
    "errors": (1606, 6, "B"), "statuses": (1612, 6, "B"),
    "temperatures": (1618, 6, "B"),
}
ERROR_BITS = ("stall", "overtemperature", "overcurrent", "motor", "communication")


class HandError(RuntimeError):
    """Base class for communication, device and motion errors."""


class HandTimeout(HandError):
    """No matching valid response; a write may nevertheless have executed."""


class HandProtocolError(HandError):
    pass


class HandMotionError(HandError):
    pass


def _integer(value: int, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _six(values: Sequence[int], high: int, *, keep: bool = False) -> list[int]:
    result = list(values)
    if len(result) != 6:
        raise ValueError("Exactly six values are required, in CHANNELS order")
    for value in result:
        _integer(value, -1 if keep else 0, high, "channel value")
    return result


def build_frame(can_id: int, data: bytes) -> bytes:
    """21-byte LOGICAL frame; call encode_wire() before serial transmission."""
    _integer(can_id, 0, 0x1FFFFFFF, "CAN ID")
    if not 1 <= len(data) <= 8:
        raise ValueError("CAN payload must contain 1..8 bytes")
    body = struct.pack("<I", can_id) + data.ljust(8, b"\xff")
    body += bytes((len(data), 0, 1, 0))  # DLC, data frame, extended ID, reserved
    return b"\xaa\xaa" + body + bytes((sum(body) & 255,)) + b"\x55\x55"


def encode_wire(frame: bytes) -> bytes:
    """Escape AA/55/A5 in body AND checksum by prefixing A5.

    Verified against the attached adapter. Wire frames can exceed 21 bytes.
    Header AA AA and footer 55 55 are not escaped.
    """
    body = bytearray()
    for value in frame[2:-2]:
        if value in (0xAA, 0x55, 0xA5):
            body.append(0xA5)
        body.append(value)
    return frame[:2] + body + frame[-2:]


class InspireHand:
    def __init__(self, port: str = "COM3", baudrate: int = 115200,
                 hand_id: int = 1, timeout: float = 0.5,
                 trace: Callable[[str, bytes], None] | None = None):
        self.port = port
        self.baudrate = _integer(baudrate, 1, 4_000_000, "baudrate")
        self.hand_id = _integer(hand_id, 1, 16383, "hand ID")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.trace = trace
        self._serial = None
        self._buffer = bytearray()
        self._lock = threading.RLock()
        self.read_retry_count = 0
        self.write_ack_recovered_count = 0
        self.communication_events: list[dict] = []

    @staticmethod
    def ports() -> list[dict]:
        return [{"port": p.device, "description": p.description, "hwid": p.hwid}
                for p in list_ports.comports()]

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def connect(self) -> "InspireHand":
        with self._lock:
            if self.is_connected:
                return self
            try:
                self._serial = serial.Serial(
                    self.port, self.baudrate, bytesize=8, parity="N", stopbits=1,
                    timeout=min(0.02, self.timeout), write_timeout=self.timeout,
                    xonxoff=False, rtscts=False, dsrdtr=False)
                self._serial.reset_input_buffer()
                self._buffer.clear()
                # Read multi-byte telemetry, not a one-byte request resembling echo.
                angles = self.read6("angles")
                if any(not 0 <= a <= 1000 for a in angles):
                    raise HandProtocolError(f"Unexpected angles: {angles}")
            except Exception:
                self.disconnect()
                raise
            return self

    def disconnect(self) -> None:
        """Close COM port. This does not stop an already commanded motion."""
        with self._lock:
            if self._serial is not None:
                self._serial.close()
                self._serial = None
            self._buffer.clear()

    close = disconnect

    def __enter__(self) -> "InspireHand":
        return self.connect()

    def __exit__(self, *_args) -> None:
        self.disconnect()

    def _next_frame(self) -> bytes | None:
        while len(self._buffer) >= 2:
            start = self._buffer.find(b"\xaa\xaa")
            if start < 0:
                self._buffer[:] = self._buffer[-1:] if self._buffer[-1] == 0xAA else b""
                return None
            del self._buffer[:start]
            cursor = 2
            body = bytearray()
            malformed = False
            while len(body) < 17:
                if cursor >= len(self._buffer):
                    return None
                value = self._buffer[cursor]
                cursor += 1
                if value == 0xA5:
                    if cursor >= len(self._buffer):
                        return None
                    value = self._buffer[cursor]
                    cursor += 1
                    if value not in (0xAA, 0x55, 0xA5):
                        malformed = True
                        break
                body.append(value)
            if malformed:
                del self._buffer[0]
                continue
            if cursor + 2 > len(self._buffer):
                return None
            frame = b"\xaa\xaa" + bytes(body) + bytes(self._buffer[cursor:cursor+2])
            if (frame[-2:] != b"\x55\x55" or frame[14] not in range(1, 9)
                    or frame[15:18] != b"\x00\x01\x00"
                    or (sum(frame[2:18]) & 255) != frame[18]):
                del self._buffer[0]
                continue
            del self._buffer[:cursor+2]
            return frame
        return None

    def _exchange(self, address: int, data: bytes, *, write: bool,
                  expected: int) -> bytes:
        _integer(address, 0, 4095, "CAN register address")
        with self._lock:
            if not self.is_connected:
                raise HandError("Not connected")
            can_id = (int(write) << 26) | (address << 14) | self.hand_id
            tx = encode_wire(build_frame(can_id, data))
            # Clear BEFORE transmit, never after: replies can arrive immediately.
            self._buffer.clear()
            self._serial.reset_input_buffer()
            try:
                sent = self._serial.write(tx)
                if sent != len(tx):
                    raise HandProtocolError("Incomplete serial write; device state is uncertain")
                if self.trace:
                    self.trace("tx", tx)
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    self._buffer.extend(self._serial.read(max(1, self._serial.in_waiting)))
                    while (frame := self._next_frame()) is not None:
                        if self.trace:
                            self.trace("rx", frame)
                        if struct.unpack_from("<I", frame, 2)[0] != can_id:
                            continue
                        if frame[14] != expected:
                            continue  # includes normal adapter request echoes
                        payload = frame[6:6 + expected]
                        # This CAN firmware returns the number of written bytes
                        # (observed 8/4/2), not the native RS485 success byte 1.
                        if write and payload != bytes((len(data),)):
                            raise HandProtocolError(f"Write rejected at {address}: {payload.hex()}")
                        return payload
            except serial.SerialException as exc:
                raise HandError(f"Serial communication failed: {exc}") from exc
            # No automatic write retry: the first command may have executed.
            raise HandTimeout(f"No valid {'write ACK' if write else 'read reply'} "
                              f"at register {address} within {self.timeout}s")

    def read_register(self, address: int, byte_count: int) -> bytes:
        """Read BYTE addresses/counts; split payload at CAN's 8-byte limit."""
        _integer(address, 0, 4095, "address")
        _integer(byte_count, 1, 4096 - address, "byte_count")
        with self._lock:
            result = bytearray()
            for offset in range(0, byte_count, 8):
                size = min(8, byte_count - offset)
                # A missing read reply was observed on the attached USB-CAN
                # adapter. Re-reading is idempotent; never replay writes.
                for attempt in range(3):
                    try:
                        reply = self._exchange(address + offset, bytes((size,)),
                                               write=False, expected=size)
                        break
                    except HandTimeout:
                        if attempt == 2:
                            raise
                        self.read_retry_count += 1
                        time.sleep(0.01)
                result.extend(reply)
            return bytes(result)

    def read6(self, name: str) -> list[int]:
        if name not in REGISTERS:
            raise ValueError(f"Unknown register group: {name}")
        address, size, fmt = REGISTERS[name]
        return list(struct.unpack("<" + fmt * 6, self.read_register(address, size)))

    def get_status(self) -> dict:
        """Sequential samples. Forces: signed g; currents: signed mA.

        Negative current values were observed during motion on this firmware;
        preserve the signed short instead of displaying them as ~65500 mA.
        """
        with self._lock:
            values = {name: self.read6(name) for name in REGISTERS}
            values["error_names"] = [
                [name for bit, name in enumerate(ERROR_BITS) if value & (1 << bit)]
                for value in values["errors"]]
            values["read_retry_count"] = self.read_retry_count
            values["write_ack_recovered_count"] = self.write_ack_recovered_count
            return values

    def _write_values(self, address: int, values: list[int]) -> None:
        data = struct.pack("<" + "h" * len(values), *values)
        for offset in range(0, len(data), 8):
            register = address + offset
            chunk = data[offset:offset + 8]
            try:
                self._exchange(register, chunk, write=True, expected=1)
            except HandTimeout as missing_ack:
                # A missing ACK does not establish that the write failed. Read
                # only the exact bytes just written; NEVER replay that write.
                event = dict(kind='write_ack_missing', address=register,
                             expected_hex=chunk.hex(), recovered=False)
                self.communication_events.append(event)
                try:
                    actual = self.read_register(register, len(chunk))
                except HandError as verification_error:
                    event['error'] = str(verification_error)
                    raise HandTimeout(
                        f'Write ACK missing at register {register}; readback failed: '
                        f'{verification_error}; write was not replayed') from missing_ack
                event['actual_hex'] = actual.hex()
                if actual != chunk:
                    raise HandProtocolError(
                        f'Write ACK missing at register {register}; readback '
                        f'{actual.hex()} != {chunk.hex()}; write was not replayed') from missing_ack
                event['recovered'] = True
                self.write_ack_recovered_count += 1

    def _set_group(self, name: str, values: Sequence[int], high: int) -> None:
        values = _six(values, high)
        with self._lock:
            self._write_values(REGISTERS[name][0], values)
            actual = self.read6(name)
            if actual != values:
                raise HandProtocolError(f"{name} readback {actual} != requested {values}")

    def set_speed(self, speeds: Sequence[int]) -> None:
        """0..1000 relative speed. Zero is NOT an emergency stop."""
        self._set_group("speeds", speeds, 1000)

    def set_force(self, forces: Sequence[int]) -> None:
        """0..3000 fingertip force threshold in grams, per supplied FTP manual."""
        self._set_group("force_limits", forces, 3000)

    def set_positions(self, positions: Sequence[int]) -> None:
        """0..1000 angle scale, -1 skips channel; 1000 opens bending channels.

        Sends individual 2-byte writes for active channels. This avoids commanding
        unrelated channels and is NOT a synchronized multi-axis trajectory.
        This low-level method does not set speed/force or wait for arrival.
        """
        values = _six(positions, 1000, keep=True)
        with self._lock:
            for channel, value in enumerate(values):
                if value != -1:
                    self._write_values(1486 + 2 * channel, [value])
            if any(value != -1 for value in values):
                actual = self.read6("targets")
                if any(v != -1 and v != a for v, a in zip(values, actual)):
                    raise HandProtocolError(f"Target readback mismatch: {actual} != {values}")

    set_angles = set_positions

    def hold(self) -> list[int]:
        """Best-effort position hold, NOT power removal or a hardware e-stop.

        Requires working communication and drives. Does not retry on failure.
        """
        with self._lock:
            angles = self.read6("angles")
            _six(angles, 1000)
            self.set_positions(angles)
            return angles

    def move(self, positions: Sequence[int], *, speed: int = 100, force: int = 300,
             timeout: float = 8.0, tolerance: int = 20,
             max_temperature: int = 55,
             cancel_event: threading.Event | None = None) -> list[int]:
        """Configure speed/force, move, and verify position with fault checks.

        Defaults are conservative commissioning settings, not a safety guarantee.
        On fault/timeout/interrupt attempts hold once and propagates the error.
        """
        values = _six(positions, 1000, keep=True)
        _integer(speed, 1, 1000, "speed")
        _integer(force, 1, 3000, "force")
        _integer(tolerance, 1, 100, "tolerance")
        _integer(max_temperature, 1, 100, "max_temperature")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                raise HandMotionError("Motion cancelled before dispatch")
            if all(v == -1 for v in values):
                return self.read6("angles")
            self._check_health(max_temperature)
            self.set_speed([speed] * 6)
            self.set_force([force] * 6)
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise HandMotionError("Motion cancelled before target write")
                self.set_positions(values)
                deadline = time.monotonic() + timeout
                actual = None
                while time.monotonic() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        raise HandMotionError("Motion cancelled; attempting position hold")
                    self._check_health(max_temperature)
                    actual = self.read6("angles")
                    if all(v == -1 or abs(v - a) <= tolerance for v, a in zip(values, actual)):
                        return actual
                    if cancel_event is not None:
                        cancel_event.wait(0.05)
                    else:
                        time.sleep(0.05)
                raise HandMotionError(f"Target not reached: target={values}, actual={actual}")
            except (Exception, KeyboardInterrupt) as exc:
                try:
                    self.hold()
                except Exception as hold_error:
                    if hasattr(exc, "add_note"):
                        exc.hold_failed = True
                        exc.add_note(f"Position hold also failed: {hold_error}")
                    else:
                        failure = HandMotionError(f"{exc}; position hold also failed: {hold_error}")
                        failure.hold_failed = True
                        if hasattr(exc, 'device_status'):
                            failure.device_status = exc.device_status
                        raise failure from exc
                raise

    def _check_health(self, max_temperature: int) -> None:
        errors = self.read6("errors")
        temperatures = self.read6("temperatures")
        if any(errors) or max(temperatures) >= max_temperature:
            failure = HandMotionError(f"Device fault/temperature: {errors=}, {temperatures=}")
            failure.device_status = {'errors': errors, 'temperatures': temperatures}
            raise failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--id", type=int, default=1)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("ports")
    subs.add_parser("status")
    subs.add_parser("hold")
    motion = subs.add_parser("move")
    motion.add_argument("angles", nargs=6, type=int)
    motion.add_argument("--speed", type=int, default=100)
    motion.add_argument("--force", type=int, default=300)
    motion.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    try:
        if args.command == "ports":
            result = InspireHand.ports()
        else:
            with InspireHand(args.port, args.baud, args.id) as hand:
                if args.command == "status":
                    result = hand.get_status()
                elif args.command == "hold":
                    result = {"held_angles": hand.hold()}
                else:
                    result = {"angles": hand.move(args.angles, speed=args.speed,
                              force=args.force, timeout=args.timeout)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (HandError, serial.SerialException, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
