import torch

def apply_rope_original(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    phases_complex = torch.polar(torch.ones_like(phases), phases)
    x_embed = torch.view_as_real(x_complex * phases_complex.unsqueeze(-2)).reshape(*x.shape[:-1], -1)
    return x_embed.type_as(x)

def apply_rope_new(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    
    cos_phases = torch.cos(phases).unsqueeze(-2)
    sin_phases = torch.sin(phases).unsqueeze(-2)
    
    out_real = x_real * cos_phases - x_imag * sin_phases
    out_imag = x_real * sin_phases + x_imag * cos_phases
    
    out_reshaped = torch.stack([out_real, out_imag], dim=-1)
    x_embed = out_reshaped.reshape(*x.shape[:-1], -1).to(x.dtype)
    return x_embed

x = torch.randn(2, 4, 16, 64) # B, S, H, C
phases = torch.randn(2, 4, 32) # B, S, C/2
out1 = apply_rope_original(x, phases)
out2 = apply_rope_new(x, phases)
print("Difference:", torch.abs(out1 - out2).max().item())
