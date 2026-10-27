"""Download, verify and restore the published P01 calibration sessions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://github.com/Jameszfyj/eeg-acquisition-decoding/releases/download/p01-bundle-20261027/'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(part, cache):
    name = part['name']
    if PurePosixPath(name).name != name or '\\' in name:
        raise ValueError('下载文件名无效')
    target = cache / name
    if target.is_file() and target.stat().st_size == part['bytes'] and sha256(target) == part['sha256']:
        return target
    temporary = target.with_name(name + '.downloading')
    for attempt in range(3):
        try:
            with urlopen(BASE + name, timeout=60) as response, temporary.open('wb') as output:
                count = 0
                while block := response.read(256 * 1024):
                    count += len(block)
                    if count > part['bytes']:
                        raise ValueError('下载大小超过清单')
                    output.write(block)
            if temporary.stat().st_size != part['bytes'] or sha256(temporary) != part['sha256']:
                raise ValueError('分卷校验不通过')
            temporary.replace(target)
            print('已校验：' + name, flush=True)
            return target
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)


def restore(archive, root):
    """Restore only calibration folders, never model or workbench preferences."""
    root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = PurePosixPath(info.filename)
            if '..' in relative.parts or relative.is_absolute() or '\\' in info.filename:
                raise ValueError('压缩包路径无效')
            if relative.parts[:2] != ('runtime', 'continuous') or info.is_dir():
                continue
            target = (root / info.filename).resolve()
            if not target.is_relative_to(root / 'runtime/continuous'):
                raise ValueError('校准数据路径越界')
            if info.file_size > 512 * 1024 * 1024:
                raise ValueError('校准文件大小异常')
            if target.exists():
                with bundle.open(info) as source:
                    expected = hashlib.file_digest(source, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(source.read()).hexdigest()
                if sha256(target) != expected:
                    raise ValueError('已有同名文件内容不同，请保留副本后处理：' + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restoring')
            with bundle.open(info) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)


def main():
    manifest = json.loads((ROOT / 'docs/releases/p01-bundle-20261027.json').read_text(encoding='utf-8-sig'))
    cache = ROOT / 'runtime/downloads/p01-bundle-20261027'
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda part: download(part, cache), manifest['parts']))
    archive = cache / manifest['asset']
    with archive.open('wb') as output:
        for path in paths:
            with path.open('rb') as source:
                shutil.copyfileobj(source, output)
    if archive.stat().st_size != manifest['bytes'] or sha256(archive) != manifest['sha256']:
        raise ValueError('完整数据包校验不通过')
    restore(archive, ROOT)
    print('两份校准会话已还原到：' + str(ROOT / 'runtime/continuous'))


if __name__ == '__main__':
    main()
