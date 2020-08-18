import numpy as np
import torch
from logger import get_logger

logger = get_logger()


class EarlyStopping:
    """Early stops the training if validation loss
        doesn't improve after a given patience.
    """

    def __init__(self, patience=7, mode="min", verbose=True, delta=1e-4):
        """
        Args:
            patience (int): How long to wait after last time
                            validation loss improved.Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.Default: False
            delta (float) : Minimum change in the monitored quantity
                            to qualify as an improvement.Default: 0
        """
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, val_score, model, path):

        if self.mode == "min":
            score = -1.0 * val_score
        else:
            score = np.copy(val_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
            self.counter = 0

    def save_checkpoint(self, new_val_score, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(
                f'Validation score changed from ({self.val_score:.6f} --> {new_val_score:.6f}).  Saving model ...')
            logger.info(f"Saving model as {path}")
        torch.save(model.state_dict(), path)
        self.val_score = new_val_score
