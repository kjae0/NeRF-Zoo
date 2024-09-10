import torch

def save_checkpoints(ckpt_path, ckpt, epoch, train_loss, val_loss=None):
    torch.save({
        'epoch': epoch,
        'checkpoint': ckpt,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, ckpt_path)
