from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch


def compare_lists(list1, list2):
    """
    Compares two lists element-wise and finds where list1[i] < list2[i].

    Prints the index and values where the condition holds,
    and returns a list of (index, list1[i], list2[i]) tuples.

    Parameters:
        list1 (list of float or int)
        list2 (list of float or int)

    Returns:
        results (list of tuples): Each tuple is (index, list1_val, list2_val)
    """
    results = []

    for i, (a, b) in enumerate(zip(list1, list2)):
        if a > b + 0.05:
            print(f"Index {i}: {a} < {b}")
            results.append((i, a, b))

    return results


device = 'cuda:1'
# different gpu's give the same result
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

task = 'super_resolution'
factor = 4
sigma = 0.1
scale = 1.0


label_root = Path(f'/dcs/pg24/u5671205/REPA/Aligned_Dps/SNR_Results_again/uniform_steps/adaptive_norm/super_resolution/label')

delta_recon_root = Path(f'/dcs/pg24/u5671205/REPA/Aligned_Dps/SNR_Results_again/uniform_steps_repa_gt/sqrt_loss_adaptive_snr/super_resolution/recon')
normal_recon_root = Path(f'/dcs/pg24/u5671205/REPA/DAPS/results/sd/imagenet/down_sampling/samples')

psnr_delta_list = []
psnr_normal_list = []

lpips_delta_list = []
lpips_normal_list = []
for idx in tqdm(range(100)):
    fname = str(idx).zfill(5)

    label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
    normal_recon = plt.imread(normal_recon_root / f'{fname}_run0000.png')[:, :, :3]

    psnr_delta = peak_signal_noise_ratio(label, delta_recon)
    psnr_normal = peak_signal_noise_ratio(label, normal_recon)

    psnr_delta_list.append(psnr_delta)
    psnr_normal_list.append(psnr_normal)

    delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
    normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
    label = torch.from_numpy(label).permute(2, 0, 1).to(device)

    delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
    normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.

    delta_d = loss_fn_vgg(delta_recon, label)
    normal_d = loss_fn_vgg(normal_recon, label)

    lpips_delta_list.append(delta_d.item())
    lpips_normal_list.append(normal_d.item())

psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)
psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)

print(f'Delta PSNR: {psnr_delta_avg}')
print(f'Delta LPIPS: {lpips_delta_avg}')

print(f'Normal PSNR: {psnr_normal_avg}')
print(f'Normal LPIPS: {lpips_normal_avg}')


compare_lists(lpips_delta_list, lpips_normal_list)