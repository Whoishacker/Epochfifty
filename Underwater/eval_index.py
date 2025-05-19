import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
from tqdm import tqdm
import os
import cv2
from Function.imqual_utils import getSSIM, getPSNR
from Function.UCIQE import calculate_uciqe
from Function.UIQM import calculate_uiqm
from basicsr.metrics import calculate_niqe


def SSIMs_PSNRs_UCIQEs_UIQMs_NIQEs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths 参考图像
        - gen_dir contain generated images 生成图像
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    pbar = tqdm(os.listdir(gen_dir))
    ssims, psnrs, uciqes, uiqms, niqes = [], [], [], [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f == gen_f):
            # assumes same filenames
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
            # 计算UCIQE, UIQM
            img = cv2.cvtColor(np.array(g_im), cv2.COLOR_RGB2BGR)

            uciqe = calculate_uciqe(img)
            uciqes.append(uciqe)

            uiqm = calculate_uiqm(img)
            uiqms.append(uiqm)

            niqe = calculate_niqe(img, crop_border=4, input_order='HWC', convert_to='y')
            niqes.append(niqe)

            pbar.update(1)
    return np.array(ssims), np.array(psnrs), np.array(uciqes), np.array(uiqms), np.array(niqes)


def main_calculate(ref_folder, out_folder):
    """计算指标主函数"""
    SSIM_measures, PSNR_measures, UCIQE_measures, UIQM_measures, NIQE_measures = SSIMs_PSNRs_UCIQEs_UIQMs_NIQEs(ref_folder, out_folder)
    print("SSIM on {0} samples".format(len(SSIM_measures))
          + f"  ==> Mean={np.mean(SSIM_measures):.7f}, Std={np.std(SSIM_measures):.7f}")
    print("PSNR on {0} samples".format(len(PSNR_measures))
          + f"  ==> Mean={np.mean(PSNR_measures):.7f}, Std={np.std(PSNR_measures):.7f}")
    print("UCIQE on {0} samples".format(len(UCIQE_measures))
          + f" ==> Mean={np.mean(UCIQE_measures):.7f}, Std={np.std(UCIQE_measures):.7f}")
    print("UIQM on {0} samples".format(len(UIQM_measures))
          + f"  ==> Mean={np.mean(UIQM_measures):.7f}, Std={np.std(UIQM_measures):.7f}")
    print("NIQE on {0} samples".format(len(NIQE_measures))
          + f"  ==> Mean={np.mean(NIQE_measures):.7f}, Std={np.std(NIQE_measures):.7f}")


if __name__=='__main__':

    ref = 'ref'
    out = 'raw-890-s'

    main_calculate(ref, out)