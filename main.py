import argparse
import sys
from trainer.train import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Custom Training Script Arguments")

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['chestexpert'],
        help="Name of the dataset to train (currently only 'chestexpert')"
    )

    parser.add_argument(
        '--loadpath',
        action='store_true',
        help="If set, model will be loaded from predefined path"
    )

    parser.add_argument(
        '--overparam',
        action='store_true',
        help="If set, use overparameterized version of the model"
    )

    parser.add_argument(
        '--depth',
        type=int,
        help="Depth of the overparameterized model (required if --overparam)"
    )

    parser.add_argument(
        '--layers',
        type=str,
        choices=['fc', 'conv', 'all'],
        help="Layers to overparameterize (required if --overparam)"
    )

    args = parser.parse_args()

    # Conditional requirement check
    if args.overparam:
        if args.depth is None or args.layers is None:
            parser.error("--depth and --layers are required when --overparam is set")

    return args

if __name__ == "__main__":
    args = get_args()
    if args.overparam:
        overparam = {'depth': args.depth, 'overparam': args.layers}
    else:
        overparam = None
    trainer = Trainer(
        dataset_name=args.dataset_name,
        loadpath=args.loadpath,
        overparam=overparam
    )
    trainer.train()