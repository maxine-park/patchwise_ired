import torch
from models import EBM, PatchEBM, DiffusionWrapper, PatchDiffusionWrapper
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, PatchGaussianDiffusion1D
import torch.nn.functional as F

def test_patch_vs_full_diffusion():
    torch.manual_seed(0)

    inp_dim = 32
    out_dim = 32
    patchsize = 32
    num_patches = out_dim // patchsize
    batch_size = 2
    timesteps = 5

    x = torch.randn(batch_size, inp_dim)
    y = torch.randn(batch_size, out_dim)
    t = torch.randint(0, timesteps, (batch_size,))

    full_model = EBM(inp_dim=inp_dim, out_dim=out_dim)
    patch_model = PatchEBM(inp_dim=inp_dim, out_dim=out_dim, patchsize=patchsize)

    with torch.no_grad():
        patch_model.patch_encoder[0].weight.copy_(full_model.fc1.weight[:, :patchsize + inp_dim])
        patch_model.patch_encoder[0].bias.copy_(full_model.fc1.bias)

    full_diff = GaussianDiffusion1D(
        model=DiffusionWrapper(full_model),
        seq_length=out_dim,
        timesteps=timesteps,
        objective='pred_noise',
    )

    patch_diff = PatchGaussianDiffusion1D(
        model=PatchDiffusionWrapper(patch_model),
        seq_length=out_dim,
        timesteps=timesteps,
        objective='pred_noise',
    )

    noise = torch.randn_like(y)
    y_noisy = full_diff.q_sample(x_start=y, t=t, noise=noise)

    t_patchwise = t.unsqueeze(1).repeat(1, num_patches)

    pred_noise_full = full_diff.model(x, y_noisy, t)
    pred_noise_patch = patch_diff.model(x, y_noisy, t_patchwise)

    diff = F.mse_loss(pred_noise_patch, pred_noise_full).item()
    max_diff = (pred_noise_patch - pred_noise_full).abs().max().item()

    print("MSE between full and patch outputs:", diff)
    print("Max abs diff:", max_diff)

    assert torch.allclose(pred_noise_patch, pred_noise_full, atol=1e-5), "Patch and full outputs differ"

if __name__ == "__main__":
    test_patch_vs_full_diffusion()
