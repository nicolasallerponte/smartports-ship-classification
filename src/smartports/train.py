import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Returns:
        Tuple of (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, leave=False, desc='  train'):
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = (torch.sigmoid(logits) >= 0.5).long()
        correct    += (preds == labels.long()).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """
    Evaluates the model on a DataLoader (val or test).

    Returns:
        Tuple of (mean_loss, accuracy, all_labels, all_probs).
        all_probs contains sigmoid probabilities for AUC-ROC computation.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    for imgs, labels in tqdm(loader, leave=False, desc='  eval '):
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = torch.sigmoid(logits)

        total_loss += loss.item() * imgs.size(0)
        preds       = (probs >= 0.5).long()
        correct    += (preds == labels.long()).sum().item()
        total      += imgs.size(0)

        all_labels.extend(labels.cpu().squeeze(1).tolist())
        all_probs.extend(probs.cpu().squeeze(1).tolist())

    return total_loss / total, correct / total, all_labels, all_probs


class EarlyStopping:
    """
    Stops training when validation loss does not improve for `patience` epochs.
    Saves the best model checkpoint.

    Args:
        patience  : Epochs to wait before stopping.
        min_delta : Minimum improvement to reset the counter.
        path      : Path to save the best model weights.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, path: str = 'best.pt'):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.best_loss = float('inf')
        self.counter   = 0
        self.stopped   = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call after each epoch.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False