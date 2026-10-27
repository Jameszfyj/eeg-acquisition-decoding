import serial


class TriggerBoxConnection:
    DEVICE_ID = 1

    FUNCTION_DEVICE_NAME = 4
    FUNCTION_OUTPUT_EVENT = 225
    FUNCTION_ERROR = 131

    ERROR_MESSAGES = {
        0: "None",
        1: "FrameHeader",
        2: "FramePayload",
        3: "ChannelNotExist",
        4: "DeviceID",
        5: "FunctionID",
        6: "SensorType",
    }

    def __init__(self):
        self._serial = None
        self._device_name = None

    @property
    def connected(self):
        return self._serial is not None and self._serial.is_open

    @property
    def device_name(self):
        return self._device_name

    def connect(self, port):
        if self.connected:
            raise RuntimeError("TriggerBox is already connected.")

        handle = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1.0,
            write_timeout=1.0,
        )

        self._serial = handle

        try:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            # The MATLAB ValidateDevice method sends GetDeviceName.
            self._send_command(
                function_id=self.FUNCTION_DEVICE_NAME,
                payload=b"",
            )

            response = self._read_response(
                expected_function_id=self.FUNCTION_DEVICE_NAME
            )

            self._device_name = response.decode(
                "ascii",
                errors="replace",
            ).rstrip("\x00")

        except Exception:
            self.disconnect()
            raise

    def send_event(self, event_code):
        if not self.connected:
            raise RuntimeError("TriggerBox is not connected.")

        if not isinstance(event_code, int):
            raise TypeError("event_code must be an integer.")

        if not 0 <= event_code <= 255:
            raise ValueError("event_code must be between 0 and 255.")

        self._send_command(
            function_id=self.FUNCTION_OUTPUT_EVENT,
            payload=bytes([event_code]),
        )

        # Successful validation means that the response header:
        # 1. Has device ID 1.
        # 2. Has function ID 225.
        # 3. Is not an error response.
        #
        # The response payload may be empty or contain device-specific data.
        return self._read_response(
            expected_function_id=self.FUNCTION_OUTPUT_EVENT
        )

    def disconnect(self):
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            finally:
                self._serial = None
                self._device_name = None

    def _send_command(self, function_id, payload):
        if not self.connected:
            raise RuntimeError("TriggerBox is not connected.")

        payload_length = len(payload)

        message = bytes([
            self.DEVICE_ID,
            function_id,
            payload_length & 0xFF,
            (payload_length >> 8) & 0xFF,
        ]) + payload

        # Remove stale responses before sending a new command.
        self._serial.reset_input_buffer()
        if self._serial.write(message) != len(message):
            raise OSError("TriggerBox serial command was not written completely.")
        self._serial.flush()

    def _read_exactly(self, size):
        data = bytearray()

        while len(data) < size:
            chunk = self._serial.read(size - len(data))

            if not chunk:
                raise TimeoutError(
                    f"Expected {size} response bytes, "
                    f"but received {len(data)}."
                )

            data.extend(chunk)

        return bytes(data)

    def _read_response(self, expected_function_id):
        # Response header:
        # byte 0: device ID
        # byte 1: function ID
        # bytes 2–3: payload length, little-endian
        header = self._read_exactly(4)

        device_id = header[0]
        function_id = header[1]
        payload_length = header[2] | (header[3] << 8)

        if device_id != self.DEVICE_ID:
            # Consume its payload before raising, keeping the stream aligned.
            if payload_length > 0:
                self._read_exactly(payload_length)

            raise RuntimeError(
                f"Unexpected device ID: {device_id}; "
                f"expected {self.DEVICE_ID}."
            )

        if function_id == self.FUNCTION_ERROR:
            error_payload = (
                self._read_exactly(payload_length)
                if payload_length > 0
                else b""
            )

            error_code = error_payload[0] if error_payload else None
            error_message = self.ERROR_MESSAGES.get(
                error_code,
                f"Undefined error code {error_code}",
            )

            raise RuntimeError(
                f"TriggerBox response error: {error_message}."
            )

        if function_id != expected_function_id:
            if payload_length > 0:
                self._read_exactly(payload_length)

            raise RuntimeError(
                f"Unexpected function ID: {function_id}; "
                f"expected {expected_function_id}."
            )

        return (
            self._read_exactly(payload_length)
            if payload_length > 0
            else b""
        )
