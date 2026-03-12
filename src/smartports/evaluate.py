import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)


# Metrics

def compute_metrics(labels: list, probs: list, threshold: float = 0.5) -> dict:
    """
    Computes all classification metrics from labels and sigmoid probabilities.

    Args:
        labels    : Ground truth binary labels.
        probs     : Sigmoid probabilities (0-1).
        threshold : Decision threshold.

    Returns:
        Dictionary with acc, f1, auc_roc, auc_pr, and confusion matrix.
    """
    preds = [1 if p >= threshold else 0 for p in probs]

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        'acc':        accuracy_score(labels, preds),
        'f1':         f1_score(labels, preds, zero_division=0),
        'auc_roc':    roc_auc_score(labels, probs),
        'auc_pr':     average_precision_score(labels, probs),
        'precision':  tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall':     tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


# Training history plots (loss + accuracy per epoch)

def plot_training_history(
    history: dict,
    fold: int,
    model_name: str,
    save_dir: Path,
) -> None:
    """
    Plots loss and accuracy curves for train and val across epochs.

    Args:
        history    : Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        fold       : Fold index (for title and filename).
        model_name : Model name string.
        save_dir   : Directory to save the figure.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'{model_name} — Fold {fold + 1} Training History', fontsize=13, fontweight='bold')

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#5b8db8', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='#e07b54', linewidth=2)
    if 'best_epoch' in history:
        axes[0].axvline(history['best_epoch'], color='green', linestyle='--', linewidth=1.5, label=f"Best epoch ({history['best_epoch']})")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', color='#5b8db8', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   label='Val',   color='#e07b54', linewidth=2)
    if 'best_epoch' in history:
        axes[1].axvline(history['best_epoch'], color='green', linestyle='--', linewidth=1.5, label=f"Best epoch ({history['best_epoch']})")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{model_name}_fold{fold + 1}_history.png', bbox_inches='tight')
    plt.close()


# Per-fold test evaluation plots

def plot_fold_results(
    labels: list,
    probs: list,
    metrics: dict,
    fold: int,
    model_name: str,
    class_names: list,
    save_dir: Path,
) -> None:
    """
    Full diagnostic plot for a single fold test evaluation:
        - Confusion matrix
        - ROC curve
        - Precision-Recall curve
        - Per-class probability distribution

    Args:
        labels      : Ground truth labels.
        probs       : Sigmoid probabilities.
        metrics     : Output of compute_metrics().
        fold        : Fold index.
        model_name  : Model name string.
        class_names : List of two class name strings e.g. ['No Ship', 'Ship'].
        save_dir    : Directory to save the figure.
    """
    preds = [1 if p >= 0.5 else 0 for p in probs]
    labels_arr = np.array(labels)
    probs_arr  = np.array(probs)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'{model_name} — Fold {fold + 1} Test Evaluation\n'
                 f'Acc={metrics["acc"]:.3f}  F1={metrics["f1"]:.3f}  '
                 f'AUC-ROC={metrics["auc_roc"]:.3f}  AUC-PR={metrics["auc_pr"]:.3f}',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.4)

    # 1. Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(class_names); ax1.set_yticklabels(class_names)
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=13)

    # 2. ROC curve
    ax2 = fig.add_subplot(gs[0, 1:3])
    fpr, tpr, _ = roc_curve(labels, probs)
    ax2.plot(fpr, tpr, color='#5b8db8', linewidth=2, label=f'AUC = {metrics["auc_roc"]:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax2.fill_between(fpr, tpr, alpha=0.1, color='#5b8db8')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Precision-Recall curve
    ax3 = fig.add_subplot(gs[0, 3])
    prec, rec, _ = precision_recall_curve(labels, probs)
    ax3.plot(rec, prec, color='#e07b54', linewidth=2, label=f'AP = {metrics["auc_pr"]:.3f}')
    ax3.fill_between(rec, prec, alpha=0.1, color='#e07b54')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Per-class probability distribution
    ax4 = fig.add_subplot(gs[1, 0:2])
    for cls, color, label in [(0, '#e07b54', class_names[0]), (1, '#5b8db8', class_names[1])]:
        ax4.hist(probs_arr[labels_arr == cls], bins=20, alpha=0.6,
                 color=color, label=label, edgecolor='white')
    ax4.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
    ax4.set_xlabel('Predicted probability')
    ax4.set_ylabel('Count')
    ax4.set_title('Probability Distribution by Class')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Metrics bar chart
    ax5 = fig.add_subplot(gs[1, 2:])
    metric_names = ['Accuracy', 'F1', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'Specificity']
    metric_vals  = [metrics['acc'], metrics['f1'], metrics['auc_roc'], metrics['auc_pr'],
                    metrics['precision'], metrics['recall'], metrics['specificity']]
    bars = ax5.barh(metric_names, metric_vals, color='#5b8db8', edgecolor='white')
    for bar, v in zip(bars, metric_vals):
        ax5.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{v:.3f}', va='center', fontsize=10)
    ax5.set_xlim(0, 1.15)
    ax5.set_title('Test Metrics')
    ax5.grid(True, alpha=0.3, axis='x')

    plt.savefig(save_dir / f'{model_name}_fold{fold + 1}_evaluation.png', bbox_inches='tight')
    plt.close()


# Summary plots across all folds

def plot_summary(
    all_metrics: list[dict],
    model_name: str,
    save_dir: Path,
) -> dict:
    """
    Aggregates metrics across folds and plots:
        - Bar chart with mean ± std per metric
        - Overlaid ROC curves for all folds

    Args:
        all_metrics : List of metrics dicts, one per fold.
        model_name  : Model name string.
        save_dir    : Directory to save the figure.

    Returns:
        Summary dict with mean and std per metric.
    """
    keys = ['acc', 'f1', 'auc_roc', 'auc_pr', 'precision', 'recall', 'specificity']
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std']  = float(np.std(vals))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — K-Fold Summary ({len(all_metrics)} folds)',
                 fontsize=13, fontweight='bold')

    # Bar chart mean ± std
    metric_names = ['Accuracy', 'F1', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'Specificity']
    means = [summary[f'{k}_mean'] for k in keys]
    stds  = [summary[f'{k}_std']  for k in keys]

    bars = axes[0].barh(metric_names, means, xerr=stds, color='#5b8db8',
                        edgecolor='white', capsize=4, error_kw={'linewidth': 1.5})
    for bar, m, s in zip(bars, means, stds):
        axes[0].text(m + s + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{m:.3f} ± {s:.3f}', va='center', fontsize=9)
    axes[0].set_xlim(0, 1.25)
    axes[0].set_title('Mean ± Std across folds')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Overlaid ROC curves per fold
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(all_metrics)))
    for i, m in enumerate(all_metrics):
        if 'fpr' in m and 'tpr' in m:
            axes[1].plot(m['fpr'], m['tpr'], color=colors[i], linewidth=1.5,
                         alpha=0.7, label=f"Fold {i+1} (AUC={m['auc_roc']:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves — All Folds')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_summary.png', bbox_inches='tight')
    plt.close()

    return summary