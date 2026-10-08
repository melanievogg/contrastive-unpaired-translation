"""Deterministic CUT diagnostics, distribution FID and persistent epoch metrics."""
import copy
import csv
import hashlib
import json
import math
import random
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

COLORS = {'train': 'tab:blue', 'val': 'tab:orange'}


def add_evaluation_options(parser, training=False):
    """Add shared deterministic evaluation settings."""
    from util.util import str2bool
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_size', type=int, default=256,
                        help='fixed square RGB resolution for all evaluation splits')
    parser.add_argument('--num_val_images', type=int, default=5)
    if training:
        parser.add_argument('--enable_validation', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--enable_fid', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--fid_freq', type=int, default=5)
        parser.add_argument('--fid_at_end', type=str2bool, nargs='?', const=True, default=True)
    return parser


def evaluation_options(opt, phase):
    """Use the existing unaligned loader with full splits and fixed resize."""
    result = copy.copy(opt)
    result.phase = phase
    result.dataset_mode = 'unaligned'
    result.serial_batches = True
    result.no_flip = True
    result.batch_size = 1
    result.num_threads = 0
    result.isTrain = False  # loader only: model remains a training CUT instance
    result.preprocess = 'resize'
    result.load_size = result.crop_size = opt.eval_size
    result.max_dataset_size = float('inf')
    return result


@contextmanager
def isolated_evaluation(model, seed):
    """Restore RNG, module modes and CUT options even if evaluation raises."""
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modules = [m for name in model.model_names for m in getattr(model, 'net' + name).modules()]
    modes = [(m, m.training) for m in modules]
    batch_size, flip = model.opt.batch_size, getattr(model.opt, 'flip_equivariance', False)
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model.opt.batch_size = 1  # PatchNCELoss holds the same opt object
        model.opt.flip_equivariance = False
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.opt.batch_size, model.opt.flip_equivariance = batch_size, flip
        for module, mode in modes:
            module.training = mode
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def loss_values(model):
    """Read the actual optimized losses without recomputing or reweighting."""
    values = model.get_current_losses()
    result = {name: values[name] for name in ('G_GAN', 'D_real', 'D_fake')}
    result['D_total'] = float(model.loss_D)
    result['NCE'] = values['NCE']
    if model.opt.nce_idt and model.opt.lambda_NCE > 0:
        result['NCE_Y'] = float(model.loss_NCE_Y)
    result['G_total'] = float(model.loss_G)
    return result


class LossMean:
    """Accumulate means weighted by actual processed image count."""
    def __init__(self):
        self.total, self.count = {}, 0

    def add(self, model, count):
        for name, value in loss_values(model).items():
            self.total[name] = self.total.get(name, 0.) + value * count
        self.count += count

    def mean(self):
        if not self.count:
            raise ValueError('No images processed; check split sizes and training batch_size.')
        return {name: value / self.count for name, value in self.total.items()}


def validation_losses(model, loader, seed):
    """Reuse original CUT losses with initialized G/D/F, without backward or steps."""
    net_f = model.netF.module if isinstance(model.netF, torch.nn.DataParallel) else model.netF
    if model.opt.lambda_NCE > 0 and getattr(net_f, 'use_mlp', False) and not net_f.mlp_init:
        raise RuntimeError('Initialize netF from a training batch before validation.')
    means = LossMean()
    with isolated_evaluation(model, seed):
        for data in loader:
            model.set_input(data)
            model.forward()  # opt.isTrain stays True, enabling identity NCE
            model.compute_D_loss()
            model.compute_G_loss()
            means.add(model, data['A'].size(0))
    return means.mean()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_epoch(path, epoch, values):
    """Keep historical epochs; replace only a rerun of the same epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_rows(path) if int(row['epoch']) != epoch]
    fields = ['epoch'] + list(values)
    if rows and set(rows[0]) != set(fields):
        raise ValueError('Metric schema changed; use a new experiment name: %s' % path)
    rows.append(dict(epoch=epoch, **values))
    rows.sort(key=lambda row: int(row['epoch']))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def rgb_tensor(tensor):
    """Convert CUT [-1,1] to RGB [0,1], identically for real and fake."""
    value = tensor.detach().float().clamp(-1, 1).add(1).div(2)
    if value.size(1) == 1:
        value = value.repeat(1, 3, 1, 1)
    if value.size(1) != 3:
        raise ValueError('FID supports only RGB or single-channel images.')
    return value


def save_image(tensor, path):
    array = rgb_tensor(tensor)[0].cpu().permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(path)


class FIDEvaluator:
    """2048-dimensional pytorch-fid features; each domain image visited once."""
    def __init__(self, opt, device, root):
        from pytorch_fid.inception import InceptionV3
        self.opt, self.device, self.root = opt, device, Path(root)
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    def features(self, tensor):
        return self.inception(rgb_tensor(tensor).to(self.device))[0].flatten(1).cpu().numpy().astype(np.float64)

    @staticmethod
    def statistics(features):
        features = np.concatenate(features, axis=0)
        if len(features) < 2:
            raise ValueError('FID needs at least two images in EACH domain.')
        if not np.isfinite(features).all():
            raise ValueError('Non-finite Inception features.')
        return features.mean(0), np.cov(features, rowvar=False)

    def split(self, model, loader, phase, examples=None, generated_dir=None):
        """Compare independent generated source and real target distributions."""
        from pytorch_fid.fid_score import calculate_frechet_distance
        dataset = loader.dataset
        forward = self.opt.direction == 'AtoB'
        source_paths = dataset.A_paths if forward else dataset.B_paths
        target_paths = dataset.B_paths if forward else dataset.A_paths
        if min(len(source_paths), len(target_paths)) < 2:
            raise ValueError('%s FID requires at least two images per domain' % phase)
        if min(len(source_paths), len(target_paths)) < 2048:
            warnings.warn('%s FID has fewer than 2048 images per domain; estimates may be unstable.' % phase)
        signature = [(str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in target_paths]
        try:
            from importlib.metadata import version
        except ImportError:  # original Python 3.6 environment
            from importlib_metadata import version
        key = hashlib.sha256(json.dumps([signature, self.opt.eval_size, self.opt.direction,
                                        version('pytorch-fid'), 'rgb-clamp-resize-v1']).encode()).hexdigest()
        cache = self.root / 'metrics' / 'fid_cache' / (key + '.npz')
        cached = cache.exists()
        real, fake = [], []
        for directory in (examples, generated_dir):
            if directory is not None:
                Path(directory).mkdir(parents=True, exist_ok=True)
        with isolated_evaluation(model, self.opt.eval_seed):
            for index, data in enumerate(loader):
                model.set_input(data)
                if not cached and index < len(target_paths):
                    real.append(self.features(model.real_B))
                if index < len(source_paths):
                    output = model.netG(model.real_A)
                    fake.append(self.features(output))
                    if examples is not None and index < self.opt.num_val_images:
                        save_image(torch.cat((model.real_A, output), dim=3), Path(examples) / ('%05d.png' % index))
                    if generated_dir is not None:
                        save_image(output, Path(generated_dir) / ('%06d_%s.png' % (index, Path(source_paths[index]).stem)))
        if cached:
            with np.load(cache) as stats:
                mu_real, sigma_real = stats['mu'], stats['sigma']
        else:
            mu_real, sigma_real = self.statistics(real)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, mu=mu_real, sigma=sigma_real)
        mu_fake, sigma_fake = self.statistics(fake)
        score = float(calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake))
        if not math.isfinite(score):
            raise ValueError('Non-finite FID for %s' % phase)
        return score


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def finish_plot(plt, root, name, ylabel, title, xlabel="Epoch"):
    root = Path(root) / 'plots'
    root.mkdir(parents=True, exist_ok=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / (name + '.png'), dpi=300)
    plt.savefig(root / (name + '.pdf'))
    plt.close()


def plot_metrics(root):
    """Refresh headless PNG/PDF plots using only recorded measurements."""
    root = Path(root)
    rows = {split: read_rows(root / 'metrics' / (split + '_losses.csv')) for split in COLORS}
    plt = pyplot()
    specifications = [('G_total', 'generator_loss', 'Generator Loss'),
                      ('D_total', 'discriminator_loss', 'Discriminator Loss'),
                      ('G_GAN', 'gan_loss', 'Adversarial Generator Loss'),
                      ('NCE', 'nce_loss', 'PatchNCE Loss'),
                      ('NCE_Y', 'nce_identity_loss', 'Identity PatchNCE Loss')]
    for field, name, title in specifications:
        if not any(data and field in data[0] for data in rows.values()):
            continue
        plt.figure()
        for split, data in rows.items():
            if data and field in data[0]:
                plt.plot([int(r['epoch']) for r in data], [float(r[field]) for r in data], label=split, color=COLORS[split])
        finish_plot(plt, root, name, title, title)
    plt.figure()
    for split, data in rows.items():
        for field, style in [('D_real', '-'), ('D_fake', '--')]:
            if data:
                plt.plot([int(r['epoch']) for r in data], [float(r[field]) for r in data], style, color=COLORS[split], label=split + ' ' + field)
    finish_plot(plt, root, 'discriminator_real_fake', 'Discriminator Loss', 'Discriminator Real/Fake Losses')
    fid = read_rows(root / 'metrics' / 'fid_scores.csv')
    if fid:
        plt.figure()
        for split in COLORS:
            plt.plot([int(r['epoch']) for r in fid], [float(r[split + '_fid']) for r in fid], 'o-', color=COLORS[split], label=split)
        best_path = root / 'metrics' / 'best_model.json'
        if best_path.exists():
            best = json.loads(best_path.read_text())
            plt.scatter([best['best_epoch']], [best['best_val_fid']], marker='*', s=150, color='red',
                        label='Best Validation FID: %.2f (Epoch %d)' % (best['best_val_fid'], best['best_epoch']))
        finish_plot(plt, root, 'fid_train_val', 'Fréchet Inception Distance (FID)', 'Training / Validation FID')


def plot_final(root, scores):
    plt = pyplot()
    plt.figure()
    labels = ['Training', 'Validation', 'Test']
    plt.bar(labels, [scores[s + '_fid'] for s in ('train', 'val', 'test')], color=['tab:blue', 'tab:orange', 'tab:green'], label='best checkpoint')
    finish_plot(plt, root, 'fid_final_comparison', 'Fréchet Inception Distance (FID)', 'Final FID — best checkpoint', xlabel='Split')


def fid_due(opt, epoch):
    return epoch % opt.fid_freq == 0 or (opt.fid_at_end and epoch == opt.n_epochs + opt.n_epochs_decay)


class TrainingEvaluation:
    """Epoch diagnostics and best validation-FID selection, never loading test."""
    def __init__(self, opt):
        from data import create_dataset
        self.opt, self.root = opt, Path(opt.checkpoints_dir) / opt.name
        if (opt.enable_validation or opt.enable_fid) and opt.model != 'cut':
            raise ValueError('CUT evaluation requires --model cut')
        if opt.phase != 'train':
            raise ValueError('Training requires --phase train; val/test must not be optimized.')
        if opt.fid_freq < 1 or opt.eval_size < 4 or opt.eval_size % 4 or opt.num_val_images < 0:
            raise ValueError('fid_freq >= 1, eval_size a positive multiple of 4, num_val_images >= 0 required.')
        self.val = create_dataset(evaluation_options(opt, 'val')) if opt.enable_validation or opt.enable_fid else None
        self.train = create_dataset(evaluation_options(opt, 'train')) if opt.enable_fid else None
        self.fid = None
        self.best = {'best_epoch': None, 'best_val_fid': float('inf')}
        best_path = self.root / 'metrics' / 'best_model.json'
        metrics = self.root / 'metrics'
        if not opt.continue_train and any((metrics / name).exists() for name in ('train_losses.csv', 'val_losses.csv', 'fid_scores.csv', 'best_model.json')):
            raise ValueError('Existing metrics: use --continue_train with --epoch_count or a new --name.')
        if opt.continue_train and best_path.exists():
            self.best = json.loads(best_path.read_text())
            if not (self.root / 'best_net_G.pth').exists():
                raise FileNotFoundError('best_model.json exists but best_net_G.pth is missing')

    def after_epoch(self, model, epoch, losses):
        write_epoch(self.root / 'metrics' / 'train_losses.csv', epoch, losses)
        if self.opt.enable_validation:
            write_epoch(self.root / 'metrics' / 'val_losses.csv', epoch,
                        validation_losses(model, self.val, self.opt.eval_seed))
        if self.opt.enable_fid and fid_due(self.opt, epoch):
            # Inception construction can consume RNG: isolate it as well.
            with isolated_evaluation(model, self.opt.eval_seed):
                if self.fid is None:
                    self.fid = FIDEvaluator(self.opt, model.device, self.root)
                train_fid = self.fid.split(model, self.train, 'train')
                val_fid = self.fid.split(model, self.val, 'val',
                                         examples=self.root / 'validation_images' / ('epoch_%d' % epoch))
            write_epoch(self.root / 'metrics' / 'fid_scores.csv', epoch, dict(train_fid=train_fid, val_fid=val_fid))
            if val_fid < self.best['best_val_fid']:
                model.save_networks('best')
                self.best = dict(best_epoch=epoch, best_val_fid=val_fid)
                write_json(self.root / 'metrics' / 'best_model.json', self.best)
        plot_metrics(self.root)
