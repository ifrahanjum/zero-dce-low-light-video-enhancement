import torch
import torch.nn.functional as F
from .model import apply_curves, NUM_ITERS

W_ILLUMINATION = 200
W_COLOR = 5
W_EXPOSURE = 10
W_SPATIAL = 1
MEAN_VAL = 0.6

def color_constancy_loss(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[2, 3])          
    mr, mg, mb = mean[:, 0], mean[:, 1], mean[:, 2]
    d_rg = (mr - mg) ** 2
    d_rb = (mr - mb) ** 2
    d_gb = (mb - mg) ** 2
    loss = torch.sqrt(d_rg**2 + d_rb**2 + d_gb**2)
    return loss.mean()

def exposure_loss(enhanced: torch.Tensor, original: torch.Tensor, mean_val: float = MEAN_VAL, patch_size: int = 16) -> torch.Tensor:
    gray_enh = enhanced.mean(dim=1, keepdim=True)   
    gray_orig = original.mean(dim=1, keepdim=True)
    mean_patch_enh = F.avg_pool2d(gray_enh, kernel_size=patch_size, stride=patch_size)
    mean_patch_orig = F.avg_pool2d(gray_orig, kernel_size=patch_size, stride=patch_size)
    mask = (mean_patch_orig < 0.6).float()
    loss = ((mean_patch_enh - mean_val) ** 2) * mask
    return loss.mean()

def illumination_smoothness_loss(curve_maps: torch.Tensor) -> torch.Tensor:
    B, C, H, W = curve_maps.shape
    h_tv = ((curve_maps[:, :, 1:, :] - curve_maps[:, :, :H-1, :]) ** 2).sum()
    w_tv = ((curve_maps[:, :, :, 1:] - curve_maps[:, :, :, :W-1]) ** 2).sum()
    count_h = W * C
    count_w = (W - 1) * C
    return 2 * (h_tv / count_h + w_tv / count_w) / B

def spatial_consistency_loss(enhanced: torch.Tensor, original: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    enh_gray  = enhanced.mean(dim=1, keepdim=True)
    orig_gray = original.mean(dim=1, keepdim=True)
    enh_pool  = F.avg_pool2d(enh_gray,  kernel_size=patch_size, stride=patch_size)
    orig_pool = F.avg_pool2d(orig_gray, kernel_size=patch_size, stride=patch_size)

    def make_kernel(dy, dx):
        k = torch.zeros(1, 1, 3, 3, device=enhanced.device)
        k[0, 0, 1, 1] = 1
        k[0, 0, 1+dy, 1+dx] = -1
        return k

    kernels = {"left": make_kernel(0, -1), "right": make_kernel(0, 1), "up": make_kernel(-1, 0), "down": make_kernel(1, 0)}
    loss = torch.zeros(1, device=enhanced.device)
    for k in kernels.values():
        d_enh  = F.conv2d(enh_pool,  k, padding=1)
        d_orig = F.conv2d(orig_pool, k, padding=1)
        loss   = loss + ((d_orig - d_enh) ** 2).mean()
    return loss

def compute_losses(image: torch.Tensor, curve_maps: torch.Tensor, num_iters: int = NUM_ITERS) -> dict:
    enhanced = apply_curves(image, curve_maps, num_iters)
    l_illumination = W_ILLUMINATION * illumination_smoothness_loss(curve_maps)
    l_spatial      = W_SPATIAL      * spatial_consistency_loss(enhanced, image)
    l_color        = W_COLOR        * color_constancy_loss(enhanced)
    l_exposure     = W_EXPOSURE     * exposure_loss(enhanced, image)
    total = l_illumination + l_spatial + l_color + l_exposure
    return {
        "total_loss": total, "illumination_smoothness_loss": l_illumination,
        "spatial_constancy_loss": l_spatial, "color_constancy_loss": l_color, "exposure_loss": l_exposure
    }