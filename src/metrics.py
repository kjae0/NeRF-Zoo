import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1: torch.Tensor, the first image (ground truth), shape (B, C, H, W) or (C, H, W)
        img2: torch.Tensor, the second image (reconstructed), shape (B, C, H, W) or (C, H, W)
        max_pixel_value: Maximum possible pixel value (1.0 if images are normalized between 0 and 1)
    
    Returns:
        psnr: Peak Signal-to-Noise Ratio (in dB)
    """
    # Ensure the inputs are float tensors
    img1 = img1.float()
    img2 = img2.float()
    
    # Compute Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')  # PSNR is infinite if MSE is 0
    
    # Compute PSNR
    psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
    
    return psnr

def calculate_mse(img1, img2):
    img1 = img1.float()
    img2 = img2.float()
    
    # Compute Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)
    
    return mse


