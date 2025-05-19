####################
# 2025-3-28
# Author: Epochfifty
# Project 4 UIE
####################
import os
import cv2
import torch
import datetime
import argparse
import threading
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import loss_igdl
from dataset import MYDataSet
from torchinfo import summary
from tensorboardX import SummaryWriter
from utils import img2tensor,tensor2img
from eval_index import main_calculate
from net01 import GeneratorNet, DiscrimiterNet
from torch.utils.data import Dataset, DataLoader



def ToTensor(image):
    """Convert ndarrays in sample to Tensors."""
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    # Normalize image from [0, 255] to [0, 1]
    image = 1 / 255.0 * image
    return torch.from_numpy(image).type(dtype=torch.float)


def plot_image(saved_d_losses, saved_g_losses):
    """绘制 dloss 和 gloss 图像"""
    # 绘制 d_loss 图表
    colors_d = plt.cm.viridis(np.linspace(0, 1, len(saved_d_losses)))
    plt.figure(figsize=(12, 6))
    for i, d_losses in enumerate(saved_d_losses):
        plt.plot(d_losses, label=f'D Loss (Epoch {i * 10})', color=colors_d[i])
    plt.xlabel('Iterations within the epoch')
    plt.ylabel('D Loss')
    plt.title('Discriminator Losses at every 10 epochs')
    plt.legend()
    d_loss_plot_path = os.path.join(log_dir, 'd_loss_plot.png')
    plt.savefig(d_loss_plot_path)
    print(f"D Loss plot saved to {d_loss_plot_path}")
    plt.close()

    # 绘制 g_loss 图表
    # colors_g = plt.cm.plasma(np.linspace(0, 1, len(saved_g_losses)))
    colors_g = plt.cm.viridis(np.linspace(0, 1, len(saved_g_losses)))
    plt.figure(figsize=(12, 6))
    for i, g_losses in enumerate(saved_g_losses):
        plt.plot(g_losses, label=f'G Loss (Epoch {i * 10})', color=colors_g[i])
    plt.xlabel('Iterations within the epoch')
    plt.ylabel('G Loss')
    plt.title('Generator Losses at every 10 epochs')
    plt.legend()
    g_loss_plot_path = os.path.join(log_dir, 'g_loss_plot.png')
    plt.savefig(g_loss_plot_path)
    print(f"G Loss plot saved to {g_loss_plot_path}")
    plt.close()


def eval_folder(NET, img_folder, checkpoint, output_folder):
    """测试模型，输出图像"""
    with torch.no_grad():
        checkpoint = torch.load(checkpoint)
        netG.load_state_dict(checkpoint)
        pbar = tqdm(os.listdir(img_folder))
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder,img_name)
            img = cv2.imread(img_path)
            high, width, _ = img.shape
            img = cv2.resize(img,(512,512))

            img_tensor = img2tensor(img)
            output_tensor = NET.forward(img_tensor)
            output_img = tensor2img(output_tensor)
            output_img = cv2.resize(output_img, (width, high))

            save_folder = output_folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder,img_name)
            cv2.imwrite(save_path,output_img)
            pbar.update(1)


parser = argparse.ArgumentParser()
# 4 train
parser.add_argument('--trainA_path', type=str, default='./data/trainA')
parser.add_argument('--trainB_path', type=str, default='./data/trainB')
parser.add_argument('--use_wgan', type=bool, default=True, help='Use WGAN to train')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=100, help='Max epoch for training')
parser.add_argument('--bz', type=int, default=32, help='batch size for training')
parser.add_argument('--lbda1', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--lbda2', type=int, default=1, help='weight for image gradient difference loss')
parser.add_argument('--num_workers', type=int, default=4, help='Use multiple kernels to load dataset')
parser.add_argument('--checkpoints_root', type=str, default='log', help='The root path to save checkpoints')
parser.add_argument('--log_root', type=str, default='log', help='The root path to save log files which are writtern by tensorboardX')
parser.add_argument('--gpu_id', type=str, default='0', help='Choose one gpu to use. Only single gpu training is supported currently')
# 4 test
parser.add_argument('--img_folder', type=str, default='raw-890-s', help='input image path')
parser.add_argument('--output_folder', type=str, default='output', help='output folder')
parser.add_argument('--reference_folder', type=str, default='ref')
parser.add_argument('--eval', type=int, default=1, help='eval model yes or no')
args = parser.parse_args()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    wgan = args.use_wgan
    learnint_rate = args.lr
    max_epoch = args.max_epoch
    batch_size = args.bz
    lambda_1 = args.lbda1
    lambda_2 = args.lbda2 # Weight for image gradient difference loss

    netG = GeneratorNet().cuda()
    summary(netG, (32, 3, 256, 256))
    netD = DiscrimiterNet(wgan_loss=wgan).cuda()

    optimizer_g = optim.Adam(netG.parameters(),lr=learnint_rate)
    optimizer_d = optim.Adam(netD.parameters(),lr=learnint_rate)
    # 数据集图像需转为256*256
    dataset = MYDataSet(src_data_path=args.trainA_path,dst_data_path=args.trainB_path)
    datasetloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=args.num_workers)

    log_root = args.log_root
    date = datetime.datetime.now().strftime('%F_%T').replace(':','_')
    log_folder = date
    log_dir = os.path.join(log_root,log_folder)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_root = args.checkpoints_root
    checkpoint_folder = date
    checkpoint_dir = os.path.join(checkpoint_root,checkpoint_folder)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_root = checkpoint_dir + '/netG_%d.pth' % (max_epoch-1)
    print(model_root)

    print("===start training===")
    total_start_time = datetime.datetime.now()

    # 用于存储每 10 轮的 d_loss_log 和 g_loss_log
    saved_d_losses = []
    saved_g_losses = []

    for epoch in range(0,max_epoch):
        d_loss_log_list = []
        g_loss_log_list = []
        start_time = datetime.datetime.now()
        for iteration, data in enumerate(datasetloader):
            batchtensor_A = data[0].cuda()
            batchtensor_B = data[1].cuda()
            generated_batchtensor = netG.forward(batchtensor_A)
            ######################
            # (1) Train Discriminator
            ######################
            num_critic = 1
            if wgan:
                num_critic = 5
            for i in range(num_critic):
                optimizer_d.zero_grad()
                d_fake = netD(generated_batchtensor)
                d_real = netD(batchtensor_B)

                #------------------------------#    
                #--- wgan loss cost function---#
                d_loss = torch.mean(d_fake) - torch.mean(d_real) # E[D(I_C)] = E[D(G(I_D))]
                
                lambda_gp = 10 # as setted in the paper
                
                epsilon = torch.rand(batchtensor_B.size()[0], 1, 1, 1).cuda()
                x_hat = batchtensor_B * epsilon + (1 -epsilon)*generated_batchtensor
                d_hat = netD.forward(x_hat)
                
                # Following code is taken from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
                # to calculate gradients penalty
                grad_outputs = torch.ones(d_hat.size()).cuda()
                gradients = torch.autograd.grad( # Calculate gradients of probabilities with respect to examples
                    outputs=d_hat,
                    inputs=x_hat,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                # Gradients have shape (batch_size, num_channels, img_width, img_height),
                # so flatten to easily take norm per example in batch
                gradients = gradients.view(batch_size,-1)
                
                # Derivatives of the gradient close to 0 can cause problems because of
                # the square root, so manually calculate norm and add epsilon
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

                # Calculate gradient penalty
                gradient_penalty = lambda_gp*torch.mean((gradients_norm - 1) ** 2)

                d_loss += gradient_penalty 
                #--- wgan loss cost function---#
                #------------------------------#   

                d_loss.backward(retain_graph=True)
                netG.zero_grad()
                optimizer_d.step()
                d_loss_log = d_loss.item()
                d_loss_log_list.append(d_loss_log)
                d_loss_log_list.append(d_loss_log)

            ######################
            # (2) Train G network
            ######################
            optimizer_g.zero_grad()
            d_fake = netD(generated_batchtensor)
            
            g_loss = -torch.mean(d_fake)
            base_loss_log = g_loss.item()
            l1_loss =  torch.mean(torch.abs(generated_batchtensor-batchtensor_B))
            l1_loss_log = l1_loss.item()
            # 计算梯度损失
            igdl_loss = loss_igdl(batchtensor_B,generated_batchtensor)
            igdl_loss_log = igdl_loss.item()

            g_loss += lambda_1 * l1_loss + lambda_2 * igdl_loss
            g_loss += lambda_1 * l1_loss
            g_loss_log = g_loss.item()
            g_loss_log_list.append(g_loss_log)
            g_loss_log_list.append(g_loss_log)

            g_loss.backward()
            netD.zero_grad()
            optimizer_g.step()

            writer.add_scalar('G_loss',g_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('D_loss',d_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('base_loss',base_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('l1_loss',l1_loss_log,(epoch*len(datasetloader)+iteration))
            writer.add_scalar('IGDL_loss',igdl_loss_log,(epoch*len(datasetloader)+iteration))
            if iteration % 10 == 0:
                end_time = datetime.datetime.now()
                time_difference = end_time - start_time

                # 获取总秒数
                total_seconds = time_difference.total_seconds()

                # 计算分钟和剩余秒数
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60

                print('==>Epoch:%d/%d:%02d d_loss:%.3f g_loss:%.3f time:%d:%.7f'
                      % (epoch, max_epoch, iteration, d_loss_log, g_loss_log, minutes, seconds))
                start_time = datetime.datetime.now()

        d_loss_average_log = np.array(d_loss_log_list).mean()
        g_loss_average_log = np.array(g_loss_log_list).mean()

        writer.add_scalar('D_loss_epoch',d_loss_average_log,epoch)
        writer.add_scalar('G_loss_epoch',g_loss_average_log,epoch)

        # 保存模型
        if epoch == (max_epoch-1):
            torch.save(netD.state_dict(), os.path.join(checkpoint_dir, 'netD_%d.pth' % epoch))
            torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'netG_%d.pth' % epoch))
            saved_d_losses.append(d_loss_log_list)
            saved_g_losses.append(g_loss_log_list)
        if epoch % 10 == 0:
            saved_d_losses.append(d_loss_log_list)
            saved_g_losses.append(g_loss_log_list)

    writer.close()

    # 保存损失数据到文件
    np.save(os.path.join(log_dir, 'saved_d_losses.npy'), saved_d_losses)
    np.save(os.path.join(log_dir, 'saved_g_losses.npy'), saved_g_losses)
    print(f"d_loss_log 数据已保存到 {os.path.join(log_dir, 'saved_d_losses.npy')}")
    print(f"g_loss_log 数据已保存到 {os.path.join(log_dir, 'saved_g_losses.npy')}")

    plot_image(saved_d_losses, saved_g_losses)

    total_end_time = datetime.datetime.now()
    total_time_difference = total_end_time - total_start_time

    total_end_time_test_difference = datetime.timedelta(0)
    total_end_time_cal_difference = datetime.timedelta(0)

    eval_bool = args.eval
    if eval_bool:
        # 根据模型，输出增强图像
        eval_folder(NET=netG,
                    img_folder=args.img_folder,
                    checkpoint=model_root,
                    output_folder=args.output_folder)
        total_end_time_test = datetime.datetime.now()
        total_end_time_test_difference = total_end_time_test - total_end_time
        # 将增强图像与参考GT图像进行对比，计算指标
        main_calculate(args.reference_folder, args.output_folder)
        total_end_time_cal = datetime.datetime.now()
        total_end_time_cal_difference = total_end_time_cal - total_end_time_test

    print("Time for train:::::::::" + str(total_time_difference))
    print("Time for test::::::::::" + str(total_end_time_test_difference))
    print("Time for calculation:::" + str(total_end_time_cal_difference))
    print("Time for all procedure:" + str(total_time_difference +
                                          total_end_time_test_difference +
                                          total_end_time_cal_difference))