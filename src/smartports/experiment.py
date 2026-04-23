import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve

from smartports.dataset import SmartportsDataset
from smartports.transforms import get_transforms
from smartports.models import get_model, unfreeze_backbone
from smartports.train import train_one_epoch, evaluate_loader, EarlyStopping
from smartports.evaluate import compute_metrics, plot_training_history, plot_fold_results, plot_summary, plot_mean_std_history


# Config

import multiprocessing

def _get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE     = _get_device()
PIN_MEMORY = isinstance(DEVICE, torch.device) and DEVICE.type == 'cuda'
# num_workers > 0 on Windows causes worker processes to re-import this module,
# which triggers device init spam and multiprocessing instability.
NUM_WORKERS = 0 if multiprocessing.get_start_method(allow_none=True) in (None, 'spawn') else 2

if __name__ == '__main__' or multiprocessing.current_process().name == 'MainProcess':
    print(f'Device     : {DEVICE}')

DEFAULT_CFG = {
    'n_folds':        5,
    'max_epochs':     50,
    'patience':       7,
    'batch_size':     16,
    'lr':             1e-3,
    'lr_resnet':      1e-4,
    'lr_transfer':    1e-4,
    'unfreeze_epoch': 5,
    'random_state':   42,
}


# Single fold

def run_fold(
    fold:        int,
    train_idx:   np.ndarray,
    val_idx:     np.ndarray,
    test_idx:    np.ndarray,
    dataset_cfg: dict,
    model_name:  str,
    augment:     bool,
    cfg:         dict,
    figures_dir: Path,
    ckpt_dir:    Path,
    save_plots:  bool = True,
) -> tuple[dict, dict]:
    """
    Trains and evaluates a single fold.

    Returns:
        Tuple of (metrics dict, history dict).
    """
    print(f'\n  Fold {fold + 1} | model={model_name} | augment={augment}')
    print(f'  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

    train_ds = SmartportsDataset(
        csv_path  = dataset_cfg['csv_path'],
        img_dir   = dataset_cfg['img_dir'],
        task      = dataset_cfg['task'],
        indices   = train_idx.tolist(),
        transform = get_transforms(augment=augment),
    )
    val_ds = SmartportsDataset(
        csv_path  = dataset_cfg['csv_path'],
        img_dir   = dataset_cfg['img_dir'],
        task      = dataset_cfg['task'],
        indices   = val_idx.tolist(),
        transform = get_transforms(augment=False),
    )
    test_ds = SmartportsDataset(
        csv_path  = dataset_cfg['csv_path'],
        img_dir   = dataset_cfg['img_dir'],
        task      = dataset_cfg['task'],
        indices   = test_idx.tolist(),
        transform = get_transforms(augment=False),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    is_resnet = model_name == 'resnet18'
    model     = get_model(model_name, freeze_backbone=is_resnet).to(DEVICE)

    lr        = cfg['lr_resnet'] if is_resnet else cfg['lr']
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    pos_weight = torch.tensor([dataset_cfg.get('pos_weight', 1.0)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    task           = dataset_cfg['task']
    ckpt_path      = ckpt_dir / f'{model_name}_aug{augment}_{task}_fold{fold + 1}.pt'
    early_stopping = EarlyStopping(patience=cfg['patience'], path=str(ckpt_path))

    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_epoch = 1

    for epoch in range(1, cfg['max_epochs'] + 1):
        if is_resnet and epoch == cfg['unfreeze_epoch']:
            unfreeze_backbone(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr_resnet'])
            print(f'  → Backbone unfrozen at epoch {epoch}')

        train_loss, train_acc   = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'  Epoch {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc:.3f} '
              f'| val_loss={val_loss:.4f} acc={val_acc:.3f}')

        if early_stopping.step(val_loss, model):
            best_epoch = max(1, epoch - cfg['patience'])
            print(f'  Early stopping at epoch {epoch}, best={best_epoch}')
            break
        else:
            if val_loss <= early_stopping.best_loss:
                best_epoch = epoch

    history['best_epoch'] = best_epoch

    # Save history to disk
    exp_name     = f'{model_name}_aug{augment}_{dataset_cfg["task"]}'
    history_path = ckpt_dir / f'{exp_name}_fold{fold + 1}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    # Evaluate test with best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    _, _, test_labels, test_probs = evaluate_loader(model, test_loader, criterion, DEVICE)

    metrics        = compute_metrics(test_labels, test_probs)
    fpr, tpr, _    = roc_curve(test_labels, test_probs)
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()

    task        = dataset_cfg['task']
    class_names = ['No Ship', 'Ship'] if task == 'ship' else ['Undocked', 'Docked']

    if save_plots:
        plot_training_history(history, fold, exp_name, figures_dir)
        plot_fold_results(test_labels, test_probs, metrics, fold, exp_name, class_names, figures_dir)

    return metrics, history


# Full K-Fold experiment

def run_experiment(
    csv_path:   Path,
    img_dir:    Path,
    task:       str,
    model_name: str,
    augment:    bool,
    pos_weight: float = 1.0,
    cfg:        dict  = DEFAULT_CFG,
    output_dir: Path  = Path('outputs'),
    save_plots: bool  = True,
) -> pd.DataFrame:
    """
    Runs full K-Fold cross-validation for a given model, task, and augmentation setting.
    """
    figures_dir = output_dir / 'figures'
    results_dir = output_dir / 'results'
    ckpt_dir    = output_dir / 'checkpoints'
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = {
        'csv_path':   csv_path,
        'img_dir':    img_dir,
        'task':       task,
        'pos_weight': pos_weight,
    }

    full_ds = SmartportsDataset(csv_path=csv_path, img_dir=img_dir, task=task)
    labels  = np.array(full_ds.get_labels())
    indices = np.arange(len(labels))

    kfold = StratifiedKFold(
        n_splits     = cfg['n_folds'],
        shuffle      = True,
        random_state = cfg['random_state'],
    )

    exp_name      = f'{model_name}_aug{augment}_{task}'
    all_metrics   = []
    all_histories = []

    print(f'\n{"="*60}')
    print(f'Experiment : {exp_name}')
    print(f'Device     : {DEVICE}')
    print(f'Folds      : {cfg["n_folds"]}  |  Max epochs: {cfg["max_epochs"]}')
    print(f'{"="*60}')

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(indices, labels)):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size    = 0.25,
            stratify     = labels[train_val_idx],
            random_state = cfg['random_state'],
        )

        metrics, history = run_fold(
            fold        = fold,
            train_idx   = train_idx,
            val_idx     = val_idx,
            test_idx    = test_idx,
            dataset_cfg = dataset_cfg,
            model_name  = model_name,
            augment     = augment,
            cfg         = cfg,
            figures_dir = figures_dir,
            ckpt_dir    = ckpt_dir,
            save_plots  = save_plots,
        )
        all_metrics.append(metrics)
        all_histories.append(history)

    # Summary
    metric_keys = ['acc', 'f1', 'auc_roc', 'auc_pr', 'precision', 'recall', 'specificity']
    if save_plots:
        summary = plot_summary(all_metrics, exp_name, figures_dir)
        plot_mean_std_history(all_histories, exp_name, figures_dir)
    else:
        summary = {}
        for k in metric_keys:
            vals = [m[k] for m in all_metrics]
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std']  = float(np.std(vals))

    # Results dataframe
    rows = []
    for i, m in enumerate(all_metrics):
        row = {'fold': i + 1}
        row.update({k: round(m[k], 4) for k in metric_keys})
        rows.append(row)

    rows.append({'fold': 'mean', **{k: round(summary[f'{k}_mean'], 4) for k in metric_keys}})
    rows.append({'fold': 'std',  **{k: round(summary[f'{k}_std'],  4) for k in metric_keys}})

    df      = pd.DataFrame(rows)
    csv_out = results_dir / f'{exp_name}_results.csv'
    df.to_csv(csv_out, index=False)

    print(f'\nResults saved to {csv_out}')
    print(df.to_string(index=False))

    return df


# Transfer learning helpers

def _reset_head(model: nn.Module, model_name: str) -> None:
    """Reinitialises the classification head with Xavier/zeros init."""
    if model_name == 'resnet18':
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
    elif model_name == 'simplecnn':
        head = model.classifier[-1]
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)


# Transfer fold

def run_fold_transfer(
    fold:            int,
    train_idx:       np.ndarray,
    val_idx:         np.ndarray,
    test_idx:        np.ndarray,
    dataset_cfg:     dict,
    model_name:      str,
    augment:         bool,
    cfg:             dict,
    figures_dir:     Path,
    ckpt_dir:        Path,
    source_ckpt_dir: Path,
    save_plots:      bool = True,
) -> tuple[dict, dict]:
    """
    Fine-tunes a model pre-trained on ship/no-ship for the docked/undocked task.

    Loads the ship checkpoint for the same fold, resets the classification head,
    then fine-tunes all layers on the docked dataset.

    Args:
        source_ckpt_dir: Directory containing the ship-task checkpoints.
    """
    task = dataset_cfg['task']
    exp_name = f'{model_name}_transfer_aug{augment}_{task}'

    print(f'\n  Fold {fold + 1} | model={model_name}_transfer | augment={augment}')
    print(f'  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

    train_ds = SmartportsDataset(
        csv_path  = dataset_cfg['csv_path'],
        img_dir   = dataset_cfg['img_dir'],
        task      = task,
        indices   = train_idx.tolist(),
        transform = get_transforms(augment=augment),
    )
    val_ds = SmartportsDataset(
        csv_path  = dataset_cfg['csv_path'],
        img_dir   = dataset_cfg['img_dir'],
        task      = task,
        indices   = val_idx.tolist(),
        transform = get_transforms(augment=False),
    )
    test_ds = SmartportsDataset(
        csv_path  = dataset_cfg['csv_path'],
        img_dir   = dataset_cfg['img_dir'],
        task      = task,
        indices   = test_idx.tolist(),
        transform = get_transforms(augment=False),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Load architecture and ship weights
    model = get_model(model_name, freeze_backbone=False)
    source_ckpt = source_ckpt_dir / f'{model_name}_aug{augment}_ship_fold{fold + 1}.pt'
    if not source_ckpt.exists():
        raise FileNotFoundError(
            f'Ship checkpoint not found: {source_ckpt}\n'
            f'Run run_ship.py first to generate ship checkpoints.'
        )
    model.load_state_dict(torch.load(source_ckpt, map_location='cpu', weights_only=False))
    _reset_head(model, model_name)
    model = model.to(DEVICE)

    print(f'  → Loaded ship checkpoint: {source_ckpt.name}')

    lr        = cfg.get('lr_transfer', 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    pos_weight = torch.tensor([dataset_cfg.get('pos_weight', 1.0)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_path      = ckpt_dir / f'{model_name}_transfer_aug{augment}_{task}_fold{fold + 1}.pt'
    early_stopping = EarlyStopping(patience=cfg['patience'], path=str(ckpt_path))

    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_epoch = 1

    for epoch in range(1, cfg['max_epochs'] + 1):
        train_loss, train_acc   = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'  Epoch {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc:.3f} '
              f'| val_loss={val_loss:.4f} acc={val_acc:.3f}')

        if early_stopping.step(val_loss, model):
            best_epoch = max(1, epoch - cfg['patience'])
            print(f'  Early stopping at epoch {epoch}, best={best_epoch}')
            break
        else:
            if val_loss <= early_stopping.best_loss:
                best_epoch = epoch

    history['best_epoch'] = best_epoch

    history_path = ckpt_dir / f'{exp_name}_fold{fold + 1}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    _, _, test_labels, test_probs = evaluate_loader(model, test_loader, criterion, DEVICE)

    metrics        = compute_metrics(test_labels, test_probs)
    fpr, tpr, _    = roc_curve(test_labels, test_probs)
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()

    class_names = ['Undocked', 'Docked']

    if save_plots:
        plot_training_history(history, fold, exp_name, figures_dir)
        plot_fold_results(test_labels, test_probs, metrics, fold, exp_name, class_names, figures_dir)

    return metrics, history


# Transfer K-Fold experiment

def run_experiment_transfer(
    csv_path:        Path,
    img_dir:         Path,
    model_name:      str,
    augment:         bool,
    source_ckpt_dir: Path,
    pos_weight:      float = 1.0,
    cfg:             dict  = DEFAULT_CFG,
    output_dir:      Path  = Path('outputs'),
    save_plots:      bool  = True,
) -> pd.DataFrame:
    """
    Runs K-Fold cross-validation for docked/undocked using ship-pretrained weights.

    Args:
        source_ckpt_dir: Directory containing ship task checkpoints (must include task in name).
    """
    figures_dir = output_dir / 'figures'
    results_dir = output_dir / 'results'
    ckpt_dir    = output_dir / 'checkpoints'
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = {
        'csv_path':   csv_path,
        'img_dir':    img_dir,
        'task':       'docked',
        'pos_weight': pos_weight,
    }

    full_ds = SmartportsDataset(csv_path=csv_path, img_dir=img_dir, task='docked')
    labels  = np.array(full_ds.get_labels())
    indices = np.arange(len(labels))

    kfold = StratifiedKFold(
        n_splits     = cfg['n_folds'],
        shuffle      = True,
        random_state = cfg['random_state'],
    )

    exp_name      = f'{model_name}_transfer_aug{augment}_docked'
    all_metrics   = []
    all_histories = []

    print(f'\n{"="*60}')
    print(f'Experiment : {exp_name}')
    print(f'Device     : {DEVICE}')
    print(f'Folds      : {cfg["n_folds"]}  |  Max epochs: {cfg["max_epochs"]}')
    print(f'{"="*60}')

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(indices, labels)):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size    = 0.25,
            stratify     = labels[train_val_idx],
            random_state = cfg['random_state'],
        )

        metrics, history = run_fold_transfer(
            fold            = fold,
            train_idx       = train_idx,
            val_idx         = val_idx,
            test_idx        = test_idx,
            dataset_cfg     = dataset_cfg,
            model_name      = model_name,
            augment         = augment,
            cfg             = cfg,
            figures_dir     = figures_dir,
            ckpt_dir        = ckpt_dir,
            source_ckpt_dir = source_ckpt_dir,
            save_plots      = save_plots,
        )
        all_metrics.append(metrics)
        all_histories.append(history)

    metric_keys = ['acc', 'f1', 'auc_roc', 'auc_pr', 'precision', 'recall', 'specificity']
    if save_plots:
        summary = plot_summary(all_metrics, exp_name, figures_dir)
        plot_mean_std_history(all_histories, exp_name, figures_dir)
    else:
        summary = {}
        for k in metric_keys:
            vals = [m[k] for m in all_metrics]
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std']  = float(np.std(vals))

    rows = []
    for i, m in enumerate(all_metrics):
        row = {'fold': i + 1}
        row.update({k: round(m[k], 4) for k in metric_keys})
        rows.append(row)

    rows.append({'fold': 'mean', **{k: round(summary[f'{k}_mean'], 4) for k in metric_keys}})
    rows.append({'fold': 'std',  **{k: round(summary[f'{k}_std'],  4) for k in metric_keys}})

    df      = pd.DataFrame(rows)
    csv_out = results_dir / f'{exp_name}_results.csv'
    df.to_csv(csv_out, index=False)

    print(f'\nResults saved to {csv_out}')
    print(df.to_string(index=False))

    return df