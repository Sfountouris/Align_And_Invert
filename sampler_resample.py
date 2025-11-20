import torch
import numpy as np


from tqdm import tqdm
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x

def pixel_optimization(measurement, x_prime, operator_fn, eps=0.5 * 1e-2, max_iters=500, lpips_fn = None, img_gt = None):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-4) # Initializing optimizer
        measurement = 2 * measurement.detach() - 1 # Need to detach for weird PyTorch reasons

        # Training loop
        for itr in range(max_iters):
            optimizer.zero_grad()
            
            measurement_loss = loss(measurement, operator_fn.forward(opt_var)) 
            
            measurement_loss.backward() # Take GD step
            optimizer.step()
            
            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                break


        return opt_var


def latent_optimization(measurement, z_init, operator_fn, vae, eps=1e-3, max_iters=500, lr=None):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(z_init.device)
        latents_bias = -torch.tensor(
                    [0., 0., 0., 0.,]
                    ).view(1, 4, 1, 1).to(z_init.device)


        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        if lr is None:
            lr_val = 5e-3
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr_val) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        init_loss = 0
        losses = []
        
        for itr in range(max_iters):
            optimizer.zero_grad()
            pixel_image = vae.decode((z_init.to(torch.float32) - latents_bias) / latents_scale).sample
            pixel_image = (pixel_image + 1) / 2
            output = loss(measurement, 0.5 * operator_fn.forward(2 * pixel_image - 1) + 0.5)          

            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy() 
            
            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)
            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                break


        return z_init, init_loss




def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError
    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var
    best_denoiser = (vt + score * sigma_t * d_sigma_t) / d_alpha_t # this comes from equations 3 and 5 in the SIT paper
    return score, best_denoiser

def compute_diffusion(t_cur):
    return 2 * t_cur

def extract_prefix(string):
    return string.split('_repa')[0]


def stochastic_resample(pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = x_t.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t * pseudo_x0 + (1 - a_t) ** 2 * x_t)/(sigma + (1 - a_t) ** 2) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t) ** 2))


def euler_sampler(vae,
        dino_encoder,
        encoder_type,
        img_gt,
        model,
        latents,
        y,
        measurement,
        mask,
        l_repa,
        learning_rate,
        operator,
        num_steps=20,
        path_type="linear", # not used, just for compatability
        max_iters = 150
        ):
    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)

    x_next = latents.to(torch.float64)
    device = x_next.device
    latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
    latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
    _dtype = latents.dtype
    bound = num_steps - 101
    for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-2], t_steps[1:-1])), 
                                   total=len(t_steps) - 1, desc="Processing time steps", leave=False, unit="step"):
        x_cur = x_next.detach().clone().requires_grad_(True).to(device)
        model_input = x_cur
        y_cur = y
        kwargs = dict(y=y_cur)
        time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
        d_cur, latents_projected = model(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            )
        d_cur = d_cur.to(torch.float64)
        x_next = x_cur + (t_next - t_cur) * d_cur
        _, best_denoiser = get_score_from_velocity(d_cur, model_input, time_input, path_type=path_type)
        best_denoiser = best_denoiser.to(torch.float32)
        
        latents_projected = latents_projected[0].to(torch.float64)
        pixel_image = vae.decode((best_denoiser - latents_bias) / latents_scale).sample
        pixel_image = (pixel_image + 1) / 2
        
        if torch.all(mask == 1):
            compare = 0.5 * operator.forward(2 * pixel_image - 1) + 0.5
        else:
            compare = 0.5 * operator.forward(2 * pixel_image - 1, mask = mask) + 0.5
        difference = measurement - compare
        
        norm = torch.linalg.norm(difference)
        if torch.all(mask == 1) == False:
            projected = (2 * img_gt - 1) * mask + (1 - mask) * (2 * pixel_image - 1)
            projected = 0.5 * projected + 0.5
            representation_1 = dino_encoder.forward_features(preprocess_raw_image(projected, encoder_type))['x_norm_patchtokens']
        else:
            representation_1 = dino_encoder.forward_features(preprocess_raw_image(measurement, encoder_type))['x_norm_patchtokens']
      
             
        representation_2 = latents_projected
        repa_similarity = torch.nn.functional.cosine_similarity(representation_1, representation_2, dim  = 2)
        
        snr = t_cur / (1 - t_cur + 1e-12)    
        loss = norm + l_repa * (256 - torch.sum(repa_similarity)) # add losses over all patches


        
        norm_grad = torch.autograd.grad(loss, inputs = x_cur)[0]
        
        x_next = x_next.detach() - learning_rate * norm_grad / max(snr, 1)
        
       
        a_cur = 1 - t_cur
        a_prev = 1 - t_next
        sigma = 80 * (1 - a_prev ** 2)/(1 - a_cur ** 2) * (1 - a_cur ** 2/ a_prev ** 2) # Change the 40 value for each task

        if i > bound and i % 10 == 0:
            pseudo_x0, _ = latent_optimization(measurement=measurement, vae = vae,
                                                             z_init=best_denoiser.detach(),
                                                             operator_fn=operator, max_iters = max_iters)
            x_next = stochastic_resample(pseudo_x0=pseudo_x0, x_t=x_next, a_t=a_cur, sigma=sigma)
            pseudo_image = vae.decode((pseudo_x0 - latents_bias) / latents_scale).sample
            pseudo_image = (pseudo_image + 1) / 2
        
    pseudo_x0, _ = latent_optimization(measurement=measurement, vae = vae,
                                                             z_init=best_denoiser.detach(),
                                                             operator_fn=operator, max_iters = max_iters)
    last = vae.decode((pseudo_x0.to(torch.float32) - latents_bias) / latents_scale).sample
    if torch.all(mask == 1) == False:
        samples = (2 * img_gt - 1) * mask + (1 - mask) * last
    else:
        samples = last
    samples = 0.5 * samples + 0.5
    return pseudo_x0

