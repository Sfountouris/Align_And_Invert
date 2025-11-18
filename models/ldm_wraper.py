import torch
import torch.nn as nn
from util.wraper_functions import get_named_beta_schedule, extract_into_tensor
import numpy as np

class DiTWithVAE(nn.Module):
    def __init__(self, dit_model, vae, num_time_steps = 1000, device = '', latents_scale = [0.18215, 0.18215, 0.18215, 0.18215], latent_bias = [0.0, 0.0, 0.0, 0.0], custom_betas = None, noise_schedule = 'linear'):
        super().__init__()
        self.dit = dit_model
        self.vae = vae
        self.num_timesteps = num_time_steps
        self.device = device
        self.latents_scale = torch.tensor(latents_scale).view(1, 4, 1, 1).to(device)
        self.latents_bias = -torch.tensor(latent_bias).view(1, 4, 1, 1).to(device)
        if custom_betas is not None:
            betas = custom_betas
        else:
            betas = get_named_beta_schedule(noise_schedule, num_time_steps)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype),
            self.alphas_cumprod[:-1]
        ])
        self.timestep_map = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
             self.timestep_map.append(i)
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        # assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        # self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        # self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        if self.posterior_variance.shape[0] > 1:
            self.posterior_log_variance_clipped = torch.log(
                torch.cat([
                    self.posterior_variance[1:2],
                    self.posterior_variance[1:]
                ], dim=0)
            )
        else:
            self.posterior_log_variance_clipped = torch.tensor([], dtype=self.betas.dtype, device=device)

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def apply_model(self, x, t, c):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        new_ts = map_tensor[t]
        # print('new time step :', new_ts)
        model_kwargs = dict(y = c)
        model_output = self.dit.forward(x, new_ts, **model_kwargs)
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract_into_tensor(torch.log(self.betas), t, x.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        return model_output, model_log_variance

    def decode_first_stage(self, latents):
        with torch.no_grad():
            return self.vae.decode((latents - self.latents_bias) / self.latents_scale).sample
    
    def differentiable_decode_first_stage(self, latents):
        return self.vae.decode((latents - self.latents_bias) / self.latents_scale).sample