"""Export DOCTR OCT volumes as unpaired 2D B-scans for this CUT repository.

Run with the project's Python environment:
    .contrastive-unpaired-translation/bin/python scripts/prepare_doctr_cut.py

The existing DOCTR subject split is retained, but A/B images are placed in
separate folders. CUT's UnalignedDataset chooses B independently of A.
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


SPLIT_NAMES = ("train", "valid", "test")
DOMAIN_FOLDERS = {"A": "SELFF-OCT-preproc_v2", "B": "SD-OCT"}


def subject_splits(split_dir, source_dir):
    splits = {}
    for name in SPLIT_NAMES:
        path = split_dir / f"{name}_subjects_0.txt"
        subjects = set(path.read_text().splitlines())
        if not subjects:
            raise ValueError(f"Empty subject split: {path}")
        splits[name] = subjects
    if len(set.union(*splits.values())) != sum(map(len, splits.values())):
        raise ValueError("Subject splits overlap")
    available = {p.name for p in source_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    if set.union(*splits.values()) != available:
        raise ValueError("Subject splits do not match subjects in the source directory")
    return splits


def sample_indices(n_slices, count):
    if n_slices < count:
        raise ValueError(f"Only {n_slices} B-scans, requested {count}")
    return np.rint(np.linspace(0, n_slices - 1, count)).astype(int)


def to_png_slice(volume, index, size):
    # NIfTI shape is (A-scans, axial depth, B-scans). The transpose displays
    # axial depth vertically, as in mydata/show_img.py.
    bscan = np.asarray(volume[:, :, index]).T
    if bscan.ndim != 2:
        raise ValueError(f"Expected a 2D B-scan, got {bscan.shape}")
    # DOCTR's image data is uint8. Keep its scanner-specific intensity scale;
    # do not normalize each volume independently.
    if bscan.dtype != np.uint8:
        raise ValueError(f"Expected uint8 OCT intensities, got {bscan.dtype}")
    image = Image.fromarray(bscan, mode="L")
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    # Extend edge pixels instead of inserting scanner-specific black bars.
    # This retains the B-scan's aspect ratio without a large artificial cue
    # that would make A and B easy for the discriminator to distinguish.
    left = (size - width) // 2
    right = size - width - left
    top = (size - height) // 2
    bottom = size - height - top
    padded = np.pad(np.asarray(image), ((top, bottom), (left, right)), mode="edge")
    return Image.fromarray(padded, mode="L")


def export(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the original DOCTR directory")
    splits = subject_splits(args.split_dir, source)
    records = []
    for split in SPLIT_NAMES:
        for subject in sorted(splits[split]):
            for eye in ("OD", "OS"):
                for domain, folder in DOMAIN_FOLDERS.items():
                    path = source / subject / eye / folder / "data.nii.gz"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    image = nib.load(str(path))
                    if len(image.shape) != 3:
                        raise ValueError(f"Expected 3D NIfTI: {path}: {image.shape}")
                    indices = sample_indices(image.shape[2], args.slices_per_volume)
                    dest = output / f"{split}{domain}"
                    dest.mkdir(parents=True, exist_ok=True)
                    # Reading a compressed NIfTI once is much faster than
                    # repeatedly decompressing it for individual B-scans.
                    volume = np.asanyarray(image.dataobj)
                    for sample_no, index in enumerate(indices):
                        name = f"{subject}_{eye}_{sample_no:03d}_z{index:04d}.png"
                        target = dest / name
                        if args.overwrite or not target.exists():
                            to_png_slice(volume, int(index), args.image_size).save(target)
                        records.append((split, domain, subject, eye, int(index), str(target.relative_to(output))))
                    print(f"{split}{domain}: {subject}/{eye}: {len(indices)} B-scans", flush=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "domain", "subject", "eye", "original_bscan_index", "png"))
        writer.writerows(records)
    print(f"Exported {len(records)} B-scans to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("mydata"))
    parser.add_argument("--output", type=Path, default=Path("datasets/doctr_cut_2d"))
    parser.add_argument(
        "--split-dir", type=Path,
        default=Path("mydata/code/Segmentation/nnunet/subject_split"),
    )
    parser.add_argument("--slices-per-volume", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing PNGs")
    args = parser.parse_args()
    if args.slices_per_volume < 1 or args.image_size < 4:
        parser.error("Slice count and image size must be positive")
    export(args)


if __name__ == "__main__":
    main()
