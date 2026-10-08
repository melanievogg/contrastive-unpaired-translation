"""Synthetic-only geometry and real CUT loader tests; never read patient data."""
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]
import nibabel as nib
import numpy as np
import prepare_doctr_cut_matched as exporter


class MatchedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_matched_', dir=ROOT / 'tests')
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.splits = self.root / 'splits'
        self.splits.mkdir()
        for split, subject in zip(exporter.SPLIT_NAMES, ['002', '003', '004']):
            (self.splits / f'{split}_subjects_0.txt').write_text(subject + '\n')
            (self.source / subject).mkdir(parents=True)
        for domain, shape, spacing in [('A', (12, 8, 64), (0.02, 0.02, 0.02)), ('B', (32, 32, 140), (0.01, 0.01, 0.01))]:
            folder = self.source / '002' / 'OD' / exporter.DOMAIN_FOLDERS[domain]
            folder.mkdir(parents=True)
            data = np.broadcast_to(np.arange(shape[1], dtype=np.uint8)[None, :, None], shape).copy()
            image = nib.Nifti1Image(data, np.diag([*spacing, 1]))
            image.header.set_xyzt_units('mm')
            nib.save(image, folder / 'data.nii.gz')
            if domain == 'B':
                # Retina near top: geometrical axial crop would exclude it.
                bounds = np.broadcast_to(np.linspace(2, 10, 9)[:, None, None], (9, shape[0], shape[2])).copy()
                np.save(folder / 'boundaries.npy', bounds.transpose(0, 2, 1))
        self.args = SimpleNamespace(source=self.source, output=self.root / 'output', split_dir=self.splits,
                                    subjects=['002'], max_subjects=None, eyes=['OD'], slices_per_volume=64,
                                    axial_margin_mm=0, qc_eyes=1, dry_run=False, boundary_layout='lzx',
                                    qc_bscan_index=None, ilm_layer=0, ob_rpe_layer=8)

    def tearDown(self):
        self.temp.cleanup()

    def test_export_and_cut_loader(self):
        files = list(self.source.rglob('*.*'))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        exporter.export(self.args)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        rows = list(csv.DictReader((self.args.output / 'manifest.csv').open()))
        self.assertEqual(len(rows), 128)
        self.assertEqual(len(list((self.args.output / 'qc').glob('*.png'))), 8)
        from data.unaligned_dataset import UnalignedDataset
        from torch.utils.data import DataLoader
        opt = SimpleNamespace(dataroot=str(self.args.output), phase='train', max_dataset_size=float('inf'),
                              serial_batches=False, isTrain=True, n_epochs=100, load_size=256, crop_size=256,
                              preprocess='none', no_flip=True)
        dataset = UnalignedDataset(opt)
        dataset.current_epoch = 1
        batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
        for domain in ['A', 'B']:
            self.assertEqual(tuple(batch[domain].shape), (1, 3, 8, 12))
        print('CUT synthetic batch: A/B = 1 x 3 x 8 x 12')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            exporter.export(self.args)

    def test_dry_run_no_output(self):
        self.args.dry_run = True
        exporter.export(self.args)
        self.assertFalse(self.args.output.exists())

    def test_linear_spacing_and_axial_shift(self):
        paths = [self.source / '002' / 'OD' / exporter.DOMAIN_FOLDERS[d] for d in ['A', 'B']]
        a, sa = exporter.header(paths[0] / 'data.nii.gz')
        b, sb = exporter.header(paths[1] / 'data.nii.gz')
        bounds = exporter.boundary_array(paths[1] / 'boundaries.npy', b.shape, 'auto')
        av, bv, size, scale, offset, start = exporter.matched(a, sa, b, sb, bounds, 0)
        self.assertLess(start[1], (size[1] - a.shape[1]) // 2)
        self.assertEqual(av.shape, bv.shape)
        expected = np.rint(offset[1] + (start[1] + np.arange(8)) * scale[1]).astype(np.uint8)
        np.testing.assert_array_equal(bv[0, :, 0], expected)
        transformed = (bounds - offset[1]) / scale[1] - start[1]
        self.assertGreaterEqual(transformed.min(), 0)
        self.assertLessEqual(transformed.max(), 7)

    def test_lzx_overlay_uses_z_slice_and_axial_y(self):
        # Square lateral axes, but asymmetric x/z variation exposes swaps.
        from PIL import Image
        import json
        shape = (12, 32, 12)
        stored = np.empty((9, 12, 12), dtype=float)
        for layer in range(9):
            for z in range(12):
                stored[layer, z, :] = 2 + layer + z + np.arange(12) // 4
        path = self.root / 'boundaries.npy'
        np.save(path, stored)
        canonical = exporter.boundary_array(path, shape, 'lzx')
        z = 3
        np.testing.assert_array_equal(canonical[:, :, z], stored[:, z, :])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            exporter.boundary_array(path, shape, 'auto')
        data = np.broadcast_to(np.arange(32, dtype=np.uint8)[None, :, None], shape).copy()
        image = nib.Nifti1Image(data, np.eye(4))
        exporter.save_boundary_qc(image, canonical, self.root, '002', 'OD', z, 0, 8, 'lzx')
        stem = self.root / '002_OD_SD_original_z0003'
        with Image.open(str(stem) + '_raw.png') as raw:
            np.testing.assert_array_equal(np.asarray(raw)[:, :, 0], data[:, :, z].T)
        with Image.open(str(stem) + '_boundaries.png') as overlay:
            self.assertEqual(overlay.size, (12, 32))
            for layer, color in [(0, (0, 255, 0)), (8, (255, 0, 255))]:
                self.assertEqual(overlay.getpixel((5, int(stored[layer, z, 5]))), color)
        metadata = json.loads(Path(str(stem) + '_boundaries.json').read_text())
        self.assertEqual(metadata['curves']['ILM']['y_original_voxels'], stored[0, z].tolist())

    def test_impossible_retina_fails(self):
        bounds = np.zeros((9, 32, 140))
        bounds[-1] = 31
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            exporter.plan_sd((32, 32, 140), np.ones(3)*0.01, (12, 8, 64), np.ones(3)*0.02, bounds, 0)

    def test_source_overlap_fails(self):
        self.args.output = self.source / 'export'
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            exporter.export(self.args)


if __name__ == '__main__':
    unittest.main()
