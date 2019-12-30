import os
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc
from myssim import compare_ssim as ssim

from config import config

output_dir = config.TEST.result_path

SCALE = 1

def output_psnr_mse(img_orig, img_out):
  squared_error = np.square(img_orig - img_out)
  mse = np.mean(squared_error)
  psnr = 10 * np.log10(1.0 / mse)
  return psnr

def _open_img(img_p):
  F = scipy.misc.fromimage(Image.open(img_p)).astype(float)/255.0
  h, w, _ = F.shape
  F = F[:h-h%SCALE, :w-w%SCALE, :]
  boundarypixels = SCALE
  F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
  return F

def _open_img_ssim(img_p):
  F = scipy.misc.fromimage(Image.open(img_p))
  h, w, _ = F.shape
  F = F[:h-h%SCALE, :w-w%SCALE, :]
  boundarypixels = SCALE
  F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
  return F

def compute_psnr(ref_im, res_im):
  return output_psnr_mse(
    _open_img(ref_im),
    _open_img(res_im)
  )

def compute_mssim(ref_im, res_im):
  ref_img = _open_img_ssim(ref_im)
  res_img = _open_img_ssim(res_im)
  channels = []
  for i in range(3):
    channels.append(ssim(ref_img[:,:,i],res_img[:,:,i],
    gaussian_weights=True, use_sample_covariance=False))
  return np.mean(channels)

def evaluate(ref_dir, res_dir, is_valid=True):
  ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
  res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
  scores_psnr = []
  scores_ssim = []
  for (ref_im, res_im) in zip(ref_pngs, res_pngs):
    psnr = compute_psnr(os.path.join(ref_dir, ref_im), os.path.join(res_dir, res_im))
    ssim = compute_mssim(os.path.join(ref_dir, ref_im), os.path.join(res_dir, res_im))
    scores_psnr.append(psnr)
    scores_ssim.append(ssim)
    print("HR {}, SR {}: PSNR = {:2.2f}, SSIM = {:.4f}".format(ref_im, res_im, psnr, ssim))
    
  psnr = np.mean(scores_psnr)
  mssim = np.mean(scores_ssim)
  
  if is_valid:
    with open(os.path.join(output_dir, 'scores_valid.txt'), 'w') as output_file:
      output_file.write("Average PSNR:%f\n"%psnr)
      output_file.write("Average SSIM:%f\n"%mssim)
      i = 0
      for (ref_im, res_im) in zip(ref_pngs, res_pngs):
        output_file.write("HR {}, SR {}: PSNR = {:2.2f}, SSIM = {:.4f}\n".format(ref_im, res_im, scores_psnr[i], scores_ssim[i]))
        i = i + 1
      output_file.close()
  else:
    with open(os.path.join(output_dir, 'scores_test.txt'), 'w') as output_file:
      output_file.write("Average PSNR:%f\n"%psnr)
      output_file.write("Average SSIM:%f\n"%mssim)
      i = 0
      for (ref_im, res_im) in zip(ref_pngs, res_pngs):
        output_file.write("HR {}, SR {}: PSNR = {:2.2f}, SSIM = {:.4f}\n".format(ref_im, res_im, scores_psnr[i], scores_ssim[i]))
        i = i + 1
      output_file.close()
