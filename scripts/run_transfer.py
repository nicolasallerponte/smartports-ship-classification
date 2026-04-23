import argparse
from pathlib import Path
from smartports.experiment import run_experiment_transfer, DEFAULT_CFG

DATA_DIR   = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'

CSV_PATH = DATA_DIR / 'docked.csv'
IMG_DIR  = DATA_DIR / 'images'

ALL_EXPERIMENTS = [
    ('simplecnn', False),
    ('simplecnn', True),
    ('resnet18',  False),
    ('resnet18',  True),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Docked/Undocked fine-tuning from ship-pretrained weights'
    )

    parser.add_argument('--model', type=str, default=None,
                        choices=['simplecnn', 'resnet18'],
                        help='Run only this model (default: all)')
    parser.add_argument('--augment', type=str, default=None,
                        choices=['true', 'false'],
                        help='Run only with/without augmentation (default: both)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CFG['max_epochs'],
                        help=f'Max epochs (default: {DEFAULT_CFG["max_epochs"]})')
    parser.add_argument('--folds', type=int, default=DEFAULT_CFG['n_folds'],
                        help=f'Number of folds (default: {DEFAULT_CFG["n_folds"]})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CFG['batch_size'],
                        help=f'Batch size (default: {DEFAULT_CFG["batch_size"]})')
    parser.add_argument('--patience', type=int, default=DEFAULT_CFG['patience'],
                        help=f'Early stopping patience (default: {DEFAULT_CFG["patience"]})')
    parser.add_argument('--lr-transfer', type=float, default=DEFAULT_CFG['lr_transfer'],
                        help=f'Learning rate for fine-tuning (default: {DEFAULT_CFG["lr_transfer"]})')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory')
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help='Directory with ship checkpoints (default: <output-dir>/checkpoints)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiments = ALL_EXPERIMENTS
    if args.model:
        experiments = [(m, a) for m, a in experiments if m == args.model]
    if args.augment is not None:
        aug_bool = args.augment == 'true'
        experiments = [(m, a) for m, a in experiments if a == aug_bool]

    if not experiments:
        print('No experiments match the given filters.')
        exit(1)

    output_dir = Path(args.output_dir)
    source_ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else output_dir / 'checkpoints'

    cfg = DEFAULT_CFG.copy()
    cfg['max_epochs']  = args.epochs
    cfg['n_folds']     = args.folds
    cfg['batch_size']  = args.batch_size
    cfg['patience']    = args.patience
    cfg['lr_transfer'] = args.lr_transfer

    print(f'Running {len(experiments)} transfer experiment(s): {experiments}')
    print(f'Source checkpoints: {source_ckpt_dir}')

    for model_name, augment in experiments:
        run_experiment_transfer(
            csv_path        = CSV_PATH,
            img_dir         = IMG_DIR,
            model_name      = model_name,
            augment         = augment,
            source_ckpt_dir = source_ckpt_dir,
            pos_weight      = 1.0,
            cfg             = cfg,
            output_dir      = output_dir,
            save_plots      = not args.no_plots,
        )
