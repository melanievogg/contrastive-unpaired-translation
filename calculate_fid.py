#!/usr/bin/env python3
"""Distribution-level OCT FID. See docs/fid.md for interpretation and usage."""
import argparse
from collections import Counter, defaultdict
import csv
from importlib.metadata import version
import math
import os
from pathlib import Path
import sys

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
DIMS = 2048
CSV_FIELDS = ['label', 'direction', 'real_dir', 'fake_dir', 'n_real', 'n_fake', 'fid']


def check_test_path(path):
    # Check both the supplied name and symlink destination against training splits.
    for candidate in (path.absolute(), path.resolve()):
        if any(p.lower() in {'traina', 'trainb', 'train', 'training'} for p in candidate.parts):
            raise ValueError(f'Training data are forbidden: {path}')


def inspect_images(root, name):
    from PIL import Image
    check_test_path(root)
    if not root.is_dir():
        raise ValueError(f'Not a directory: {root}')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len(files) < 2:
        raise ValueError(f'{name}: need at least two images, found {len(files)} in {root}')
    sizes, modes, groups = Counter(), Counter(), defaultdict(list)
    for path in files:
        check_test_path(path)
        with Image.open(path) as img:
            # Pillow RGB conversion clips high-bit-depth grayscale. Do not silently
            # introduce an intensity policy for scientific source images.
            if img.mode not in {'1', 'L', 'RGB', 'RGBA', 'LA', 'P'}:
                raise ValueError(f'{path}: mode {img.mode} cannot be converted without '
                                 'a potentially destructive intensity conversion; use an '
                                 'explicitly defined 8-bit evaluation dataset.')
            if getattr(img, 'n_frames', 1) != 1:
                raise ValueError(f'{path}: multi-frame images are ambiguous; supply individual B-scans.')
            img.load()  # Detect corrupt images before downloading weights / inference.
            sizes[img.size] += 1
            modes[img.mode] += 1
            groups[img.size].append(path)
    print(f'{name} images: {len(files)}\n  directory: {root.resolve()}')
    print('  first 3: ' + ', '.join(str(p.relative_to(root)) for p in files[:3]))
    print(f'  sizes (width, height): {dict(sorted(sizes.items()))}\n  Pillow modes: {dict(sorted(modes.items()))}')
    return files, groups


def statistics(groups, model, batch_size, device):
    import numpy as np
    from pytorch_fid.fid_score import get_activations
    # Upstream ImagePathDataset explicitly calls Image.open(...).convert('RGB').
    # L -> RGB replicates channels. ToTensor only converts uint8 to float / 255.
    # Batch only equal native sizes: no resizing, cropping or padding here.
    # Concatenate ALL features before computing a single covariance, never
    # average per-batch/per-size FIDs or statistics.
    activations = np.concatenate([
        get_activations(groups[size], model, min(batch_size, len(groups[size])),
                        DIMS, device, num_workers=0)
        for size in sorted(groups)
    ], axis=0)
    if not np.isfinite(activations).all():
        raise ValueError('Non-finite Inception features.')
    return activations.mean(axis=0), np.cov(activations, rowvar=False)


def validate_csv(path):
    if path.suffix.lower() != '.csv':
        raise ValueError('--output-csv must have a .csv extension (never write to an image).')
    if path.exists() and path.stat().st_size:
        with path.open(newline='', encoding='utf-8') as stream:
            if next(csv.reader(stream), None) != CSV_FIELDS:
                raise ValueError(f'Existing CSV has an incompatible header: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True, help='Original target testA/testB directory')
    parser.add_argument('--fake', type=Path, required=True, help='Generated test fake_A/fake_B directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default=None, help='cpu, cuda or cuda:0; default: CUDA if available')
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--label', default='')
    parser.add_argument('--direction', choices=['AtoB', 'BtoA'], help='Inferred from real testA/testB if omitted')
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError('--batch-size must be positive')
        check_test_path(args.real)
        check_test_path(args.fake)
        inferred = {'testA': 'BtoA', 'testB': 'AtoB'}.get(args.real.name)
        if inferred is None:
            raise ValueError('--real must be the original dataset testA or testB directory.')
        direction = args.direction or inferred
        if direction != inferred:
            raise ValueError('--direction conflicts with the real target test directory.')
        expected_fake = 'fake_B' if direction == 'AtoB' else 'fake_A'
        if args.fake.name != expected_fake:
            raise ValueError(f'{direction} requires a dedicated {expected_fake} directory; '
                             'do not supply a mixed results/images directory.')
        if args.output_csv:
            validate_csv(args.output_csv)
        real_files, real_groups = inspect_images(args.real, 'real')
        fake_files, fake_groups = inspect_images(args.fake, 'fake')
        if len(real_files) != len(fake_files):
            print('WARNING: real/fake counts differ! Using every image without truncation.', file=sys.stderr)
        print('Plausibility only: 10 subjects x 2 eyes x 64 B-scans = approximately 1280 images/domain.')
        print('Input provenance must be verified by the caller; filenames alone cannot prove test membership.')
        # Set before CUDA initialization. No random augmentation or shuffled loading.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import numpy as np
        import torch
        from pytorch_fid.inception import InceptionV3
        from pytorch_fid.fid_score import calculate_frechet_distance
        np.random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f'Device: {device}\nFID feature dimension: {DIMS}\nDirection: {direction}\nBatch size: {args.batch_size}')
        print('Versions: ' + ', '.join(f'{p}={version(p)}' for p in
                                      ['pytorch-fid', 'torch', 'torchvision', 'numpy', 'scipy', 'Pillow']))
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]],
                            resize_input=True, normalize_input=True, use_fid_inception=True).to(device).eval()
        real_mu, real_cov = statistics(real_groups, model, args.batch_size, device)
        fake_mu, fake_cov = statistics(fake_groups, model, args.batch_size, device)
        fid = float(calculate_frechet_distance(real_mu, real_cov, fake_mu, fake_cov))
        if not math.isfinite(fid):
            raise ValueError('FID is not finite; no CSV result written.')
        print(f'\nFID: {fid:.4f}\nreal images: {len(real_files)}\nfake images: {len(fake_files)}')
        if args.output_csv:
            validate_csv(args.output_csv)
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            header_needed = not args.output_csv.exists() or args.output_csv.stat().st_size == 0
            with args.output_csv.open('a', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                if header_needed:
                    writer.writerow(CSV_FIELDS)
                writer.writerow([args.label, direction, str(args.real.resolve()), str(args.fake.resolve()),
                                 len(real_files), len(fake_files), repr(fid)])
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\nDependencies: python3 -m pip install pytorch-fid==0.3.0\n')


if __name__ == '__main__':
    main()
