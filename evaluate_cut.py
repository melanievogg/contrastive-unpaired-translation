"""Final evaluation of the selected best generator on three independent splits."""
import json
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.cut_evaluation import (add_evaluation_options, evaluation_options,
                                 FIDEvaluator, isolated_evaluation, write_json, plot_final)


class EvaluationOptions(TestOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        add_evaluation_options(parser)
        parser.set_defaults(model='cut', epoch='best')
        return parser


def main():
    opt = EvaluationOptions().parse()
    if opt.model != 'cut' or opt.epoch != 'best':
        raise ValueError('Final CUT evaluation requires --model cut --epoch best')
    if opt.eval_size < 4 or opt.eval_size % 4:
        raise ValueError('--eval_size must be a positive multiple of four')
    root = Path(opt.checkpoints_dir) / opt.name
    best = json.loads((root / 'metrics' / 'best_model.json').read_text())
    # Verify ALL splits before inference/download; test never falls back to val.
    loaders = {phase: create_dataset(evaluation_options(opt, phase)) for phase in ('train', 'val', 'test')}
    model = create_model(opt)
    model.setup(opt)  # inference configuration loads only best_net_G.pth
    model.eval()
    with isolated_evaluation(model, opt.eval_seed):
        fid = FIDEvaluator(opt, model.device, root)
        scores = {phase + '_fid': fid.split(model, loader, phase,
                  generated_dir=root / 'test_images' if phase == 'test' else None)
                  for phase, loader in loaders.items()}
    result = dict(checkpoint='best', best_epoch=best['best_epoch'], **scores)
    write_json(root / 'metrics' / 'test_metrics.json', result)
    plot_final(root, scores)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
