"""Local Chinese WAV cues. No cloud TTS, extra package, or live-device access."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import wave


SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Speech
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voices = @($synth.GetInstalledVoices() | Where-Object {
    $_.Enabled -and $_.VoiceInfo.Culture.Name -eq 'zh-CN'
  })
  if ($voices.Count -eq 0) { throw 'No local zh-CN voice is installed.' }
  $synth.SelectVoice($voices[0].VoiceInfo.Name)
  $synth.Rate = 0
  foreach ($item in $request) {
    $synth.SetOutputToWaveFile([string]$item.path)
    $synth.Speak([string]$item.text)
    $synth.SetOutputToNull()
  }
  [Console]::Write($voices[0].VoiceInfo.Name)
} finally { $synth.Dispose() }
"""


class SpeechCache:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()

    @staticmethod
    def key(text):
        return hashlib.sha256(("windows-local-zh-CN-v1\n" + text).encode("utf-8")).hexdigest()

    def file(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("无效的提示音编号。")
        return self.root / (key + ".wav")

    def prepare(self, texts):
        if not isinstance(texts, list) or not 1 <= len(texts) <= 16:
            raise ValueError("提示文字数量须为1至16条。")
        if any(not isinstance(t, str) or not 1 <= len(t.strip()) <= 100 for t in texts):
            raise ValueError("每条语音文字须为1至100个字符。")
        texts = list(dict.fromkeys(t.strip() for t in texts))
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            missing = [{"text": t, "path": str(self.file(self.key(t)).with_suffix(".part.wav").resolve())}
                       for t in texts if not self.file(self.key(t)).is_file()]
            if missing:
                if os.name != "nt":
                    raise ValueError("本机语音需要Windows中文语音；可关闭声音后采集。")
                exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
                encoded = base64.b64encode(SCRIPT.encode("utf-16-le")).decode("ascii")
                try:
                    result = subprocess.run(
                        [str(exe), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                        input=json.dumps(missing, ensure_ascii=False), text=True, encoding="utf-8",
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90,
                        creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise ValueError("无法准备本机中文语音，请检查Windows中文语音或关闭声音后采集。") from exc
                if result.returncode:
                    raise ValueError("本机中文语音生成失败，请检查Windows中文语音，或关闭声音后采集。")
                for item in missing:
                    source = Path(item["path"])
                    if not source.is_file():
                        raise ValueError("提示音生成不完整，请重新试听。")
                    source.replace(self.file(self.key(item["text"])))
            cues = []
            for text in texts:
                key = self.key(text)
                try:
                    with wave.open(str(self.file(key)), "rb") as stream:
                        duration = stream.getnframes() / stream.getframerate()
                except (OSError, wave.Error) as exc:
                    raise ValueError("提示音缓存不可读取，请检查runtime/audio_cache。") from exc
                if not 0 < duration <= 35:
                    raise ValueError("单条语音过长，请缩短任务文字至约30秒内能读完。")
                cues.append({"text": text, "key": key, "url": "/audio/" + key + ".wav",
                             "duration_seconds": duration})
            return {"engine": "windows_local_zh_CN", "cues": cues}
