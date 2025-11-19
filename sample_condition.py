from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
import pickle

import torch.nn.functional as F
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
# from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from samplers import euler_sampler
from data.dataloader import get_dataset, get_dataloader

from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from util.utils import load_encoders
import numpy as np
from util.utils import load_legacy_checkpoints, download_model
from tqdm import tqdm
import random
from util.plot_functions import plot_and_save_info


def interpolant(self, t):
    if self.path_type == "linear":
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = -1
        d_sigma_t =  1
    elif self.path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError()
    return alpha_t, sigma_t, d_alpha_t, d_sigma_t


VALID_ALGORITHMS = [
    "_latent_dps.png",
    "_latent_dps_repa.png"
]



def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    # logger

    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    task_config = load_yaml(args.task_config)

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = False,
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(ckpt_path, map_location=f'{device}', weights_only = False)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    latents_scale = torch.tensor(args.latents_scale).view(1, 4, 1, 1).to(device)
    latents_bias = -torch.tensor(args.latent_bias).view(1, 4, 1, 1).to(device)

    dino_encoder, encoder_types, _ = load_encoders(args.enc_type, device)
    dino_encoder[0].eval()

    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
   
    # Working directory
    out_path = os.path.join(args.save_dir + '_' + str(seed), measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    for i, ref_img in tqdm(enumerate(loader), total=len(loader), desc="Processing Images", unit="batch"):
        ref_img = ref_img.to(device)            
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            forward = operator.forward(ref_img, mask=mask)
            y_n = noiser(forward)         
        else: 
            y = operator.forward(ref_img)
            y_n = noiser(y)
            mask = torch.ones(y_n.size()).to(device)
        if task_config['measurement']['operator']['name'] == 'super_resolution':
            y_n_temp = F.interpolate(y_n, size=(256, 256), mode='nearest')
        else:
            y_n_temp = y_n
        img_gt = (ref_img + 1) / 2
        measurement = (y_n + 1) / 2
        z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        y = torch.tensor([1000], device=device) # no guidance
        # Sample images:
        sampling_kwargs = dict(vae = vae,
            dino_encoder = dino_encoder[0],
            encoder_type = encoder_types[0],
            img_gt = img_gt,
            model=model, 
            latents=z,
            y=y,
            l_repa = args.l_repa,
            learning_rate = args.learning_rate,
            num_steps=args.num_steps, 
            path_type=args.path_type,
            measurement = measurement,
            mask = mask,
            operator = operator,
        )
        
        samples = euler_sampler(**sampling_kwargs)
        samples = samples.to(torch.float32)

        samples = vae.decode((samples -  latents_bias) / latents_scale).sample
        if measure_config['operator'] ['name'] == 'inpainting':
                samples = ref_img * mask + (1 - mask) * samples
        samples = (samples + 1) / 2.
        
        samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for j, sample in enumerate(samples):
            ref_img_plot = (ref_img + 1) / 2
            y_n_plot = (y_n_temp + 1) / 2
            if task_config['measurement']['operator']['name'] == 'super_resolution':
                y_n = F.interpolate(y_n, size=(256, 256), mode='bilinear', align_corners=False)
            ref_img_plot = torch.clamp(
                255. * ref_img_plot, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            y_n_plot = torch.clamp(
                255. * y_n_plot, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            ref_img_plot = np.squeeze(ref_img_plot)
            y_n_plot = np.squeeze(y_n_plot)
            # this function will plot inference results
            plot_and_save_info(sample, out_path, fname,ref_img_plot, y_n_plot)
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
        
    # encoder type
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    
    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument(
        "--latents-scale", 
        type=float, 
        default=[0.18215, 0.18215, 0.18215, 0.18215],
        )

    parser.add_argument(
        "--latent-bias", 
        type=float, 
        default=[0.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument("--l_repa", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=2.0)
    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode

    # all the previous are from repa these are from dps

    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='results_test_expectation/uniform_steps_repa_800')

    args = parser.parse_args()
    main(args)

