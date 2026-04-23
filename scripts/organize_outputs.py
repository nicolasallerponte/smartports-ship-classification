"""
Organizes outputs/figures into a structured directory tree:

outputs/figures/
├── eda/
│   ├── class_distribution.png
│   ├── brightness.png
│   ├── resolutions.png
│   └── samples_grid.png
├── ship/
│   ├── simplecnn/
│   │   ├── augFalse/
│   │   │   ├── fold1_history.png
│   │   │   ├── ...
│   │   │   ├── fold1_evaluation.png
│   │   │   ├── ...
│   │   │   ├── summary.png
│   │   │   └── mean_std_history.png
│   │   └── augTrue/
│   │       └── ...
│   ├── convnext/
│   └── resnet18/
└── docked/
    ├── simplecnn/
    ├── convnext/
    └── resnet18/
"""

import shutil
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / 'outputs' / 'figures'


def organize():
    if not FIGURES_DIR.exists():
        print(f'Figures dir not found: {FIGURES_DIR}')
        return

    files = list(FIGURES_DIR.glob('*.png'))
    print(f'Found {len(files)} PNG files to organize')

    moved, skipped = 0, 0

    for f in files:
        name = f.stem  # e.g. simplecnn_augFalse_ship_fold1_history

        # EDA figures
        if name in ('class_distribution', 'brightness', 'resolutions', 'samples_grid'):
            dest_dir = FIGURES_DIR / 'eda'

        # Experiment figures
        else:
            # Determine task
            if '_ship_' in name or name.endswith('_ship'):
                task = 'ship'
            elif '_docked_' in name or name.endswith('_docked'):
                task = 'docked'
            else:
                skipped += 1
                print(f'  [skip] {f.name}')
                continue

            # Determine model
            if name.startswith('simplecnn'):
                model = 'simplecnn'
            elif name.startswith('convnext'):
                model = 'convnext'
            elif name.startswith('resnet18'):
                model = 'resnet18'
            else:
                skipped += 1
                print(f'  [skip] {f.name}')
                continue

            # Determine augmentation
            if '_augFalse_' in name:
                aug = 'augFalse'
            elif '_augTrue_' in name:
                aug = 'augTrue'
            else:
                skipped += 1
                print(f'  [skip] {f.name}')
                continue

            dest_dir = FIGURES_DIR / task / model / aug

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Rename to remove redundant prefix
        # e.g. simplecnn_augFalse_ship_fold1_history.png → fold1_history.png
        # e.g. simplecnn_augFalse_ship_summary.png → summary.png
        parts = name.split('_')
        # Remove model, aug, task prefix (first 3 tokens)
        if name not in ('class_distribution', 'brightness', 'resolutions', 'samples_grid'):
            clean_name = '_'.join(parts[3:]) + f.suffix
        else:
            clean_name = f.name

        dest = dest_dir / clean_name

        if dest.exists():
            print(f'  [exists] {dest}')
            skipped += 1
            continue

        shutil.move(str(f), str(dest))
        print(f'  {f.name} → {dest.relative_to(FIGURES_DIR.parent.parent)}')
        moved += 1

    print(f'\nDone. Moved: {moved} | Skipped: {skipped}')


if __name__ == '__main__':
    organize()