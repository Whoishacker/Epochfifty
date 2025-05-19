import torch
import torch.nn as nn 
import cv2,datetime,os
from netv0_5 import GeneratorNet
import argparse
import numpy as np
from utils import img2tensor,tensor2img
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--img_folder',type=str,default='./imgs',help='input image path')
parser.add_argument('--checkpoint',type=str,default='netv0_5.pth',help='checkpoint for generator')
parser.add_argument('--output_folder',type=str,default='./imgs',help='output folder')
args = parser.parse_args()


def Enhance_images_main(input_folder, output_folder):
    netG = GeneratorNet().cuda()
    with torch.no_grad():
        checkpoint = torch.load(args.checkpoint)
        netG.load_state_dict(checkpoint)
        img_folder = input_folder
        pbar = tqdm(os.listdir(img_folder))
        total_psnr = 0
        total_ssim = 0
        total_img = 0
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_name)

            img = cv2.imread(img_path)

            high, width, _ = img.shape
            img = cv2.resize(img, (512, 512))

            img_tensor = img2tensor(img)
            output_tensor = netG.forward(img_tensor)
            output_img = tensor2img(output_tensor)
            output_img = cv2.resize(output_img, (width, high))

            save_folder = output_folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, img_name)
            cv2.imwrite(save_path, output_img)
            pbar.update(1)


if __name__ == "__main__":
   Enhance_images_main(input_folder='imgs', output_folder='imgs')