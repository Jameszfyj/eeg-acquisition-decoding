"""Local, sequential browser video uploads. Never connects to EEG or the hand."""
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

from .config import write_json


class ScreenRecordings:
    MAX_CHUNK = 16 * 1024 * 1024

    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()

    def _folder(self, recording_id):
        if not isinstance(recording_id, str) or len(recording_id) != 32:
            raise ValueError('录屏编号无效。')
        try:
            if uuid.UUID(recording_id).hex != recording_id:
                raise ValueError()
        except (ValueError, AttributeError):
            raise ValueError('录屏编号无效。')
        return self.root / recording_id

    def _read(self, recording_id, owner=None):
        folder = self._folder(recording_id)
        meta = json.loads((folder/'recording.json').read_text(encoding='utf-8'))
        if owner is not None and owner != meta['subject_window_id']:
            raise ValueError('录屏不属于当前任务窗口。')
        return folder, meta

    def open(self, request):
        owner = request.get('subject_window_id')
        if not isinstance(owner, str):
            raise ValueError('任务窗口编号无效。')
        try:
            if str(uuid.UUID(owner)) != owner:
                raise ValueError()
        except ValueError:
            raise ValueError('任务窗口编号无效。')
        mime = request.get('mime_type', '')
        container = mime.split(';')[0]
        if container not in ('video/webm', 'video/mp4'):
            raise ValueError('当前浏览器录屏格式不受支持。')
        recording_id = uuid.uuid4().hex
        folder = self._folder(recording_id)
        with self.lock:
            folder.mkdir(parents=True, exist_ok=False)
            filename = 'decode_screen.' + ('webm' if container == 'video/webm' else 'mp4')
            (folder/filename).touch(exist_ok=False)
            meta = dict(id=recording_id, subject_window_id=owner, mime_type=mime,
                        filename=filename, status='recording', complete=False,
                        started_at=time.strftime('%Y-%m-%d %H:%M:%S'), chunks=0, bytes=0,
                        has_audio=request.get('has_audio') is True,
                        capture_surface=str(request.get('capture_surface', 'unknown'))[:40])
            write_json(folder/'recording.json', meta)
        return dict(id=recording_id, path=str((folder/filename).resolve()))

    def append(self, recording_id, owner, sequence, data):
        if type(sequence) is not int or sequence < 0 or not 0 < len(data) <= self.MAX_CHUNK:
            raise ValueError('录屏分段编号或大小无效。')
        with self.lock:
            folder, meta = self._read(recording_id, owner)
            if meta['status'] != 'recording':
                raise ValueError('本次录屏已结束。')
            digest = hashlib.sha256(data).hexdigest()
            if sequence == meta['chunks']-1 and digest == meta.get('last_chunk_sha256'):
                return {'ok': True, 'chunks': meta['chunks']}
            if sequence != meta['chunks']:
                raise ValueError('录屏分段不连续，已保留先前视频。')
            # Truncate any bytes written before a previous interrupted metadata
            # commit, so a retried upload never appends the same segment twice.
            with (folder/meta['filename']).open('r+b') as video:
                video.truncate(meta['bytes'])
                video.seek(meta['bytes'])
                video.write(data)
                video.flush()
            meta.update(chunks=sequence+1, bytes=meta['bytes']+len(data), last_chunk_sha256=digest)
            write_json(folder/'recording.json', meta)
            return {'ok': True, 'chunks': meta['chunks']}

    def finish(self, request, state):
        with self.lock:
            folder, meta = self._read(request.get('recording_id'), request.get('subject_window_id', ''))
            if meta['status'] == 'recording':
                complete = (request.get('complete') is True and meta['chunks'] > 0
                            and request.get('chunks') == meta['chunks'])
                session = state.get('session') or state.get('saved_path')
                meta.update(status='saved' if complete else 'partial', complete=complete,
                            finished_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                            reason=str(request.get('reason', ''))[:500],
                            eeg_session=session if meta['subject_window_id'] in
                            (state.get('subject_window_id'), state.get('monitor_window_id')) else None)
                write_json(folder/'recording.json', meta)
            return dict(id=meta['id'], complete=meta['complete'], bytes=meta['bytes'],
                        path=str((folder/meta['filename']).resolve()),
                        download_url='/api/recording/file?recording_id='+meta['id'] if meta['bytes'] else None)

    def file(self, recording_id):
        with self.lock:
            folder, meta = self._read(recording_id)
            if not meta['bytes']:
                raise ValueError('本次录屏没有收到可保存的视频。')
            return folder/meta['filename'], meta['mime_type'].split(';')[0]
