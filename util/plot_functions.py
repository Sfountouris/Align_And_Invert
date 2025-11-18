import matplotlib.pyplot as plt
import os
from PIL import Image

import numpy as np


def plot_sampling_metrics(repa_sim, repa_sim_expectation, norm_losses, gt_losses, fname, out_path):
    """
    Plots EPA similarity, normalized losses, and ground truth losses across the sampling process.
    
    Args:
        epa_sim (list): List of EPA similarity values.
        norm_losses (list): List of normalized loss values.
        gt_losses (list): List of ground truth loss values.
        output_path (str): Path where the plot will be saved.
    """
    # Check that all input lists have the same length
    assert len(norm_losses) == len(gt_losses), "All input lists must have the same length."
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    first_list = [t[0] for i, t in enumerate(repa_sim)] 
    second_list = [t[1] for i, t in enumerate(repa_sim)]
    # Plot EPA similarity
    axes[0].plot(first_list, label='REPA Similarity Ground Truth', color='blue')
    # axes[0].plot(second_list, label='REPA Similarity-expectation', color='red')
    axes[0].set_title('REPA Similarity Across Sampling Process')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Similarity')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
    repa_sim_expectation[1:], 
    label='Similarity Between Ground Truth and Tweedie Representation', 
    color='blue'
)

    # Add horizontal line at the first element
    axes[1].axhline(
        y=repa_sim_expectation[0], 
        color='red', 
        linestyle='--', 
        label='Measurement Similarity'
    )

    # Improve title readability
    axes[1].set_title('Similarity Between Ground Truth and Tweedie Representation During Sampling')

    axes[1].set_xlabel('Sampling Steps')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].legend()
    axes[1].grid(True)


    # Plot ground truth losses
    gt_arr = np.array(gt_losses)              # shape (T, 2)
    loss_values = gt_arr[:, 0]                # 1 / norm
    snr_values  = gt_arr[:, 1]                # 1 / max(SNR, 1)
    # Clear the subplot before drawing (prevents accumulation when re-running)
    axes[2].cla()

    # Plot once per curve
    axes[2].plot(loss_values, label='Ground Truth Losses', color='red')
    axes[2].plot(snr_values,  label=r'$1/\max(\mathrm{SNR},1)$', linestyle='--')

    # Formatting
    axes[2].set_title('Ground Truth Error and SNR-based Weight Across Sampling Process')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Value')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'combined', f'optimization_metrics_{fname}'))
    plt.close(fig)

def plot_sampling_probe_metrics(psnr, lpips, fname, out_path):
    
    # Check that all input lists have the same length
    assert len(psnr) == len(lpips), "All input lists must have the same length."
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot EPA similarity
    axes[0].plot(psnr, label='PSNR', color='blue')
    axes[0].set_title('PSNR Across Sampling Process')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('PSNR')
    axes[0].legend()
    axes[0].grid(True)

    # Plot normalized losses
    axes[1].plot(lpips, label='Learned Perceptual Image Patch Similarity', color='green')
    axes[1].set_title('Learned Perceptual Image Patch Similarity Across Sampling Process')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('LPIPS')
    axes[1].legend()
    axes[1].grid(True)


    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'combined', f'optimization_probe_metrics_{fname}'))
    plt.close(fig)


def plot_sampling_metrics_grid(repa_sim, gt_losses, psnr, lpips, fname, out_path):
    """
    Plots REPA similarity, ground truth losses, PSNR, and LPIPS across the sampling process
    in a 2x2 grid.

    Args:
        repa_sim (list): REPA similarity values.
        gt_losses (list): Ground truth loss values.
        psnr (list): PSNR values.
        lpips (list): LPIPS values.
        fname (str): Filename identifier for saving the plot.
        out_path (str): Directory to save the plot in.
    """
    # Ensure input lengths match
    assert len(repa_sim) == len(gt_losses) == len(psnr) == len(lpips), "All input lists must have the same length."

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    first_list = [t[0] for i, t in enumerate(repa_sim[0])]  
    second_list = [t[0] for i, t in enumerate(repa_sim[1])]
    # Top-left: REPA Similarity
    axes[0, 0].plot(first_list, color='blue', label='Similarity-Repa')
    axes[0, 0].plot(second_list, color='red', label='Similarity')
    axes[0, 0].set_title('REPA Similarity')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Similarity')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Top-right: Ground Truth Loss
    axes[0, 1].plot(gt_losses[0], color='blue', label='Ground Truth Loss Repa')
    axes[0, 1].plot(gt_losses[1], color='red', label='Ground Truth Loss')
    axes[0, 1].set_title('Ground Truth Loss')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Bottom-left: PSNR
    axes[1, 0].plot(psnr[0], color='blue', label='PSNR Repa')
    axes[1, 0].plot(psnr[1], color='red', label='PSNR')
    axes[1, 0].set_title('PSNR')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('PSNR')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Bottom-right: LPIPS
    axes[1, 1].plot(lpips[0], color='blue', label='LPIPS Repa')
    axes[1, 1].plot(lpips[1], color='red', label='LPIPS')
    axes[1, 1].set_title('LPIPS')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('LPIPS')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.join(out_path, 'combined'), exist_ok=True)

    # Save the plot
    save_path = os.path.join(out_path, 'combined', f'optimization_metrics_grid_{fname}')
    plt.savefig(save_path)
    plt.close(fig)


def write_metrics_to_file(psnr, lpips, out_path,algorithm):
    """
    Appends PSNR and LPIPS metrics to separate files.
    
    Args:
    - psnr (float): The PSNR value to write.
    - lpips (float): The LPIPS value to write.
    - fname (str): The filename or identifier for the current image.
    """
    # Define the directories where the metrics will be stored
    psnr_dir = f"{out_path}/psnr_results"
    lpips_dir = f"{out_path}/lpips_results"
    
    # Ensure the directories exist
    os.makedirs(psnr_dir, exist_ok=True)
    os.makedirs(lpips_dir, exist_ok=True)
    
    # File paths for PSNR and LPIPS metrics
    psnr_file = os.path.join(psnr_dir, f"psnr_results{algorithm}.txt")
    lpips_file = os.path.join(lpips_dir, f"lpips_results{algorithm}.txt")
    
    # Append PSNR metric to the file
    with open(psnr_file, 'a') as psnr_f:
        psnr_f.write(f'{psnr}\n')
    
    # Append LPIPS metric to the file
    with open(lpips_file, 'a') as lpips_f:
        lpips_f.write(f'{lpips}\n')
    
def plot_and_save_info(sample, out_path, fname, ref_img_plot, y_n_plot):
    Image.fromarray(sample).save(os.path.join(out_path, 'recon', fname))
    Image.fromarray(ref_img_plot).save(os.path.join(out_path, 'label', fname))
    Image.fromarray(y_n_plot).save(os.path.join(out_path, 'input', fname))
    height, width, channels = sample.shape
    # Create a new blank image with enough width to hold all three images side by side
    combined_width = width * 3
    combined_image = Image.new('RGB', (combined_width, height))
    sample_img = Image.fromarray(sample)
    ref_img_img = Image.fromarray(ref_img_plot)
    y_n_img = Image.fromarray(y_n_plot)

    # Paste the images onto the new blank canvas
    combined_image.paste(sample_img, (0, 0))
    combined_image.paste(ref_img_img, (width, 0))
    combined_image.paste(y_n_img, (width * 2, 0))
    fname = fname[:-4] + '.png'
    # Save the combined image
    combined_dir = os.path.join(out_path, 'combined')

    # Create the directory if it does not exist
    os.makedirs(combined_dir, exist_ok=True)

    # Save the image to the specified path
    combined_image.save(os.path.join(combined_dir, fname))
  

