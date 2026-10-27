from pathlib import Path
import tempfile
import unittest
import zipfile

from examples.download_p01_data import restore


class ReleaseRestoreTests(unittest.TestCase):
    def test_restore_preserves_settings_and_refuses_changed_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'bundle.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('runtime/continuous/example/source/data.h5',b'raw')
                bundle.writestr('configs/workbench.local.json',b'not used')
            restore(archive,root)
            self.assertFalse((root/'configs/workbench.local.json').exists())
            path=root/'runtime/continuous/example/source/data.h5'
            self.assertEqual(path.read_bytes(),b'raw')
            restore(archive,root)
            path.write_bytes(b'user data')
            with self.assertRaisesRegex(ValueError,'已有同名'):
                restore(archive,root)
            self.assertEqual(path.read_bytes(),b'user data')

    def test_restore_rejects_parent_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'bundle.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('runtime/continuous/../../escape.txt',b'bad')
            with self.assertRaisesRegex(ValueError,'路径无效'):
                restore(archive,root)
            self.assertFalse((root/'escape.txt').exists())
