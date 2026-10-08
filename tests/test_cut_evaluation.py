"""Synthetic CPU checks for split isolation and unchanged CUT optimization."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from data import create_dataset
from models.cut_model import CUTModel
from options.train_options import TrainOptions
from util.cut_evaluation import (evaluation_options, validation_losses, loss_values,
    isolated_evaluation, write_epoch, read_rows, plot_metrics, plot_final,
    TrainingEvaluation, fid_due, FIDEvaluator, rgb_tensor, LossMean)


@pytest.fixture
def opt(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    for phase in ('train', 'val', 'test'):
        for domain, count in [('A', 2), ('B', 3)]:
            directory = tmp_path / 'data' / (phase + domain)
            directory.mkdir(parents=True)
            for i in range(count):
                array = np.random.RandomState(i + (domain == 'B') * 10).randint(0, 256, (36, 40, 3), dtype=np.uint8)
                Image.fromarray(array).save(directory / ('%d.png' % i))
    o = TrainOptions('--dataroot %s --checkpoints_dir %s --name synthetic --gpu_ids -1 --ngf 4 --ndf 4 --netF_nc 8 --nce_layers 0,4 --num_patches 8 --batch_size 2 --load_size 32 --crop_size 32 --eval_size 32 --num_threads 0 --netG resnet_6blocks' % (tmp_path / 'data', tmp_path / 'ckpt')).gather_options()
    o.isTrain, o.gpu_ids = True, []
    (Path(o.checkpoints_dir) / o.name).mkdir(parents=True)
    return o


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('phase', ['train', 'val', 'test'])
def test_split_loader(opt, phase):
    loader = create_dataset(evaluation_options(opt, phase))
    assert len(loader) == 3
    for data in loader:
        assert Path(data['A_paths'][0]).parent.name == phase + 'A'
        assert Path(data['B_paths'][0]).parent.name == phase + 'B'
        assert data['A'].shape == (1, 3, 32, 32)
    assert torch.equal(next(iter(loader))['A'], next(iter(loader))['A'])


def test_missing_test_never_uses_val(opt):
    import shutil
    shutil.rmtree(Path(opt.dataroot) / 'testB')
    with pytest.raises(FileNotFoundError, match='no split fallback'):
        create_dataset(evaluation_options(opt, 'test'))


@pytest.mark.parametrize('identity,flip', [(True, False), (False, True)])
def test_validation_preserves_training(opt, identity, flip):
    opt.nce_idt, opt.flip_equivariance = identity, flip
    model = CUTModel(opt)
    data = next(iter(create_dataset(opt)))
    model.data_dependent_initialize(data)
    model.set_input(data)
    model.optimize_parameters()
    losses = loss_values(model)
    assert losses['D_total'] == pytest.approx((losses['D_real'] + losses['D_fake']) / 2)
    assert losses['G_total'] == pytest.approx(losses['G_GAN'] + ((losses['NCE'] + losses['NCE_Y']) / 2 if identity else losses['NCE']))
    assert ('NCE_Y' in losses) == identity
    state = {n: copy.deepcopy(getattr(model, 'net' + n).state_dict()) for n in model.model_names}
    optimizers = [copy.deepcopy(o.state_dict()) for o in model.optimizers]
    modes = [m.training for n in model.model_names for m in getattr(model, 'net' + n).modules()]
    rng = torch.get_rng_state().clone()
    np_rng, py_rng = np.random.get_state(), random.getstate()
    loader = create_dataset(evaluation_options(opt, 'val'))
    values = validation_losses(model, loader, 123)
    assert values == validation_losses(model, loader, 123)
    for name in state:
        assert_equal(state[name], getattr(model, 'net' + name).state_dict())
    for previous, optimizer in zip(optimizers, model.optimizers):
        assert_equal(previous, optimizer.state_dict())
    assert modes == [m.training for n in model.model_names for m in getattr(model, 'net' + n).modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    assert py_rng == random.getstate()
    assert opt.batch_size == 2 and opt.flip_equivariance == flip


def test_csv_plots_and_weighting(tmp_path):
    model = SimpleNamespace(opt=SimpleNamespace(nce_idt=False, lambda_NCE=1), loss_D=2., loss_G=3.,
                            get_current_losses=lambda: dict(G_GAN=1., D_real=2., D_fake=2., NCE=2.))
    mean = LossMean()
    mean.add(model, 2)
    model.loss_G = 6.
    mean.add(model, 1)
    assert mean.mean()['G_total'] == 4.
    for split in ('train', 'val'):
        path = tmp_path / 'metrics' / (split + '_losses.csv')
        write_epoch(path, 1, mean.mean())
        write_epoch(path, 2, mean.mean())
        write_epoch(path, 2, mean.mean())
        assert len(read_rows(path)) == 2
    write_epoch(tmp_path / 'metrics/fid_scores.csv', 5, dict(train_fid=10., val_fid=11.))
    plot_metrics(tmp_path)
    plot_final(tmp_path, dict(train_fid=10., val_fid=11., test_fid=12.))
    for name in ('generator_loss', 'discriminator_loss', 'gan_loss', 'nce_loss', 'discriminator_real_fake', 'fid_train_val', 'fid_final_comparison'):
        for extension in ('png', 'pdf'):
            assert (tmp_path / 'plots' / (name + '.' + extension)).stat().st_size > 100


def test_fid_interval():
    o = SimpleNamespace(fid_freq=5, fid_at_end=True, n_epochs=11, n_epochs_decay=0)
    assert [e for e in range(1, 12) if fid_due(o, e)] == [5, 10, 11]


def test_best_selection_resume_without_test(opt, monkeypatch):
    import util.cut_evaluation as ev
    opt.enable_fid = True
    opt.n_epochs, opt.n_epochs_decay = 11, 0
    evaluator = TrainingEvaluation(opt)
    phases = []
    values = iter([12., 10., 11.])
    class FakeFID:
        def split(self, model, loader, phase, **kwargs):
            phases.append(phase)
            return next(values) if phase == 'val' else 20.
    evaluator.fid = FakeFID()
    saved = []
    model = SimpleNamespace(save_networks=lambda name: (saved.append(name), (evaluator.root / 'best_net_G.pth').touch()))
    from contextlib import nullcontext
    monkeypatch.setattr(ev, 'isolated_evaluation', lambda *a: nullcontext())
    monkeypatch.setattr(ev, 'plot_metrics', lambda *a: None)
    for epoch in (5, 10, 11):
        evaluator.after_epoch(model, epoch, dict(G_total=1.))
    assert saved == ['best', 'best']
    assert set(phases) == {'train', 'val'}
    opt.continue_train = True
    resumed = TrainingEvaluation(opt)
    assert resumed.best == dict(best_epoch=10, best_val_fid=10.)
    assert len(read_rows(evaluator.root / 'metrics/train_losses.csv')) == 3


def test_fid_unique_domains_cache_and_denormalization(opt, monkeypatch):
    from pytorch_fid import fid_score
    monkeypatch.setattr(fid_score, 'calculate_frechet_distance', lambda *args: 1.25)
    model = SimpleNamespace(opt=opt, model_names=['G'], netG=torch.nn.Identity(),
                            eval=lambda: None)
    def set_input(data):
        model.real_A, model.real_B = data['A'], data['B']
    model.set_input = set_input
    evaluator = object.__new__(FIDEvaluator)
    evaluator.opt, evaluator.root = opt, Path(opt.checkpoints_dir) / opt.name
    counts = []
    evaluator.features = lambda t: (counts.append(1) or np.array([[float(t.mean()), float(t.std())]]))
    loader = create_dataset(evaluation_options(opt, 'val'))
    assert evaluator.split(model, loader, 'val') == 1.25
    assert len(counts) == 5  # 2 source + 3 target, no modulo duplicates
    counts.clear()
    evaluator.split(model, loader, 'val')
    assert len(counts) == 2  # real feature statistics cached
    assert torch.equal(rgb_tensor(torch.tensor([[[[-1., 1.]]]])), torch.tensor([[[[0., 1.]], [[0., 1.]], [[0., 1.]]]]))


@pytest.mark.parametrize('identity,netf', [(True, 'mlp_sample'), (False, 'sample')])
def test_disabled_nce(opt, identity, netf):
    opt.lambda_NCE, opt.nce_idt, opt.netF = 0., identity, netf
    model = CUTModel(opt)
    data = next(iter(create_dataset(opt)))
    model.data_dependent_initialize(data)
    model.set_input(data)
    model.optimize_parameters()
    values = validation_losses(model, create_dataset(evaluation_options(opt, 'val')), 0)
    assert 'NCE_Y' not in values
    assert values['NCE'] == 0


def test_final_loads_best_generator(opt):
    from models import create_model
    train_model = CUTModel(opt)
    data = next(iter(create_dataset(opt)))
    train_model.data_dependent_initialize(data)
    train_model.save_networks('best')
    expected = copy.deepcopy(train_model.netG.state_dict())
    with torch.no_grad():
        for parameter in train_model.netG.parameters():
            parameter.add_(10)
    train_model.save_networks('latest')
    inference = copy.copy(opt)
    inference.isTrain, inference.epoch = False, 'best'
    model = create_model(inference)
    model.setup(inference)
    model.parallelize()  # explicit CPU on a host with CUDA must remain supported
    assert model.model_names == ['G']
    assert_equal(expected, model.netG.state_dict())


def test_real_inception_fid(opt):
    """Uses cached/downloaded official Inception weights and full 2048-D FID."""
    model = CUTModel(opt)
    loader = create_dataset(evaluation_options(opt, 'val'))
    with isolated_evaluation(model, 0):
        fid = FIDEvaluator(opt, model.device, Path(opt.checkpoints_dir) / opt.name)
        score = fid.split(model, loader, 'val')
    assert np.isfinite(score)
