"""Input safeguards and distribution aggregation; no model download needed."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('calculate_fid', Path(__file__).resolve().parents[1] / 'calculate_fid.py')
fid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid)


class FidTests(unittest.TestCase):
    def test_recursive_sizes_modes_and_rgb_replication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'testB'
            (root / 'subject').mkdir(parents=True)
            Image.new('L', (276, 200), 73).save(root / 'subject' / 'a.PNG')
            Image.new('RGB', (200, 276)).save(root / 'b.tiff')
            files, groups = fid.inspect_images(root, 'real')
            self.assertEqual(len(files), 2)
            self.assertEqual(set(groups), {(276, 200), (200, 276)})
            with Image.open(root / 'subject' / 'a.PNG') as img:
                self.assertEqual(img.convert('RGB').getpixel((0, 0)), (73, 73, 73))

    def test_training_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            training = Path(temp) / 'trainA'
            training.mkdir()
            alias = Path(temp) / 'testA'
            alias.symlink_to(training, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Training data'):
                fid.check_test_path(alias)

    def test_high_bit_depth_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(2):
                Image.new('I;16', (10, 10), 1024).save(root / f'{i}.tiff')
            with self.assertRaisesRegex(ValueError, 'intensity conversion'):
                fid.inspect_images(root, 'real')

    def test_all_size_groups_form_one_distribution(self):
        package = types.ModuleType('pytorch_fid')
        backend = types.ModuleType('pytorch_fid.fid_score')
        calls = []
        def activations(files, model, batch_size, dims, device, num_workers):
            calls.append((len(files), batch_size, dims))
            return np.asarray(files, dtype=float)
        backend.get_activations = activations
        rows = [[1, 2], [3, 7], [9, 5]]
        with patch.dict(sys.modules, {'pytorch_fid': package, 'pytorch_fid.fid_score': backend}):
            mean, cov = fid.statistics({(10, 10): rows[:2], (20, 10): rows[2:]}, None, 2, 'cpu')
        np.testing.assert_allclose(mean, np.mean(rows, axis=0))
        np.testing.assert_allclose(cov, np.cov(rows, rowvar=False))
        self.assertEqual(calls, [(2, 2, 2048), (1, 1, 2048)])

    def test_csv_header_and_image_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'results.csv'
            path.write_text('wrong,header\n')
            with self.assertRaisesRegex(ValueError, 'header'):
                fid.validate_csv(path)
            with path.open('w', newline='') as stream:
                csv.writer(stream).writerow(fid.CSV_FIELDS)
            fid.validate_csv(path)
            with self.assertRaisesRegex(ValueError, '.csv extension'):
                fid.validate_csv(Path(temp) / 'image.png')


if __name__ == '__main__':
    unittest.main()
