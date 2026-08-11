import numpy as np
import torch
import os
import joblib


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


    
def check_and_prepare_dirs(args):
    """
    Check if required directories exist and create output directories if they don't.
    """
    output_dirs = {
        "plots_dir": os.path.join("./plots/", args.d_setting),
        "checkpoints_dir": os.path.join(args.d_checkpoint_path, args.d_setting),
        "results_dir": os.path.join("./results/", args.d_setting),
    }

    for name, path in output_dirs.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")


def save_results(args, preds, trues):
    np.save(os.path.join("./results/", args.d_setting, "preds.npy"), preds)
    np.save(os.path.join("./results/", args.d_setting, "trues.npy"), trues)

def inverse_transform(args, data):
        scaler = joblib.load(os.path.join(args.d_checkpoint_path, args.d_setting, "scaler.pkl"))
        return scaler.inverse_transform(data)