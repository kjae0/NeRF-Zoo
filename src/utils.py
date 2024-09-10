import torch

def save_checkpoints(ckpt_path, model, optimizer, scheduler, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, ckpt_path)
