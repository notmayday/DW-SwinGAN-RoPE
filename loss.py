import torch
from torch.autograd import Function
import torch.nn as nn
import pickle
from focal_frequency_loss import FocalFrequencyLoss as FFL
import torch
from scipy.io import loadmat, savemat
import cv2
import numpy as np
from scipy.fftpack import dct, dctn, idctn
import utils.metrics as mt
class netLoss():

    def __init__(self, args, masked_kspace_flag=True):
        self.args = args

        mask_path = args.mask_path

        with open(mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
            self.masks = masks_dictionary['mask1']
            self.mask = torch.tensor(self.masks == 1, device=self.args.device)
            self.masked_kspace_flag = masked_kspace_flag
        self.maskNot = self.mask == 0


        self.ImL2_weights = args.loss_weights[0]
        self.ImL1_weights = args.loss_weights[1]
        self.KspaceL2_weights = args.loss_weights[2]
        self.AdverLoss_weight = args.loss_weights[3]
        self.FFLLoss_weight =args.loss_weights[4]
        self.DCTLoss_weight = args.loss_weights[5]
        self.DCT_imgLoss_weight = args.loss_weights[6]
        self.ImL2Loss = nn.MSELoss()
        self.ImL1Loss = nn.SmoothL1Loss()
        #self.AdverLoss = nn.BCELoss()
        self.AdverLoss = nn.BCEWithLogitsLoss()
        self.DCTLoss = nn.SmoothL1Loss()


        if self.masked_kspace_flag:
            self.KspaceL2Loss = nn.MSELoss(reduction='sum')
        else:
            self.KspaceL2Loss = nn.MSELoss()

    def img_space_loss(self,pred_Im,tar_Im):
        return self.ImL1Loss(pred_Im, tar_Im),self.ImL2Loss(pred_Im, tar_Im)

    def DCT_img_space_loss(self,pred_Im,tar_Im):
        DCT_imgLoss = self.ImL1Loss(pred_Im, tar_Im)+ self.ImL2Loss(pred_Im, tar_Im)
        return DCT_imgLoss
    def k_space_loss(self,pred_K,tar_K):
        if self.masked_kspace_flag:
            return self.KspaceL2Loss(pred_K, tar_K)/(torch.sum(self.maskNot)*tar_K.max())
        else:
            return self.KspaceL2Loss(pred_K, tar_K)

    def dct_space_loss(self, pred_dct):
        target = torch.zeros(pred_dct.shape)
        return self.DCTLoss(pred_dct, target)

    def gen_adver_loss(self,D_fake):
        real_ = torch.tensor(1.0).expand_as(D_fake).to(self.args.device)
        return self.AdverLoss(D_fake, real_)

    def disc_adver_loss(self, D_real, D_fake):
        real_ = torch.tensor(1.0).expand_as(D_real).to(self.args.device)
        fake_ = torch.tensor(0.0).expand_as(D_fake).to(self.args.device)
        real_loss = self.AdverLoss(D_real,real_)
        fake_loss = self.AdverLoss(D_fake,fake_)
        return real_loss,fake_loss

    def calc_PRE_loss(self, tar_Im, enh_img, masked_Kspaces):
        mask_expanded = self.mask.expand(self.args.batch_size, 2, self.args.img_size, self.args.img_size)
        ImL1, ImL2 = self.img_space_loss(tar_Im, enh_img)
        kspace_undersampled_p = FourierTransform(enh_img) #* mask_expanded
        kspace_undersampled_t = FourierTransform(tar_Im) #* mask_expanded
        KspaceL2 = self.k_space_loss(kspace_undersampled_p, kspace_undersampled_t)
        #KspaceL2 = abs(self.k_space_loss(kspace_undersampled_p, masked_Kspaces) - 0.001)
        dct_space = dct2d(enh_img)
        DCTLoss = self.dct_space_loss(dct_space)
        tv = total_variation(enh_img)
        ffl = FFL()
        fflLoss = self.FFLLoss = ffl(enh_img, tar_Im)
        fullLoss_PRE = self.KspaceL2_weights * KspaceL2 + self.DCTLoss_weight * DCTLoss + self.FFLLoss_weight*fflLoss
        return fullLoss_PRE

    def calc_gen_loss(self, ori_K, pred_Im, pred_K, tar_Im, tar_K, ORI_img, w1, w2, w3, enh_img, D_fake=None):
        ImL1,ImL2 = self.img_space_loss(pred_Im, tar_Im)
        ORIImL1, ORIImL2 = self.img_space_loss(ORI_img, tar_Im)
        mask_expanded = self.mask.expand(self.args.batch_size, 2, self.args.img_size, self.args.img_size)
        kspace_undersampled_p = FourierTransform(enh_img)*mask_expanded
        kspace_undersampled_t = FourierTransform(tar_Im)*mask_expanded
        kspace_undersampled_o = ori_K*mask_expanded*mask_expanded
        KspaceL2 = self.k_space_loss(kspace_undersampled_p, kspace_undersampled_t)
        DCT_imgLoss = self.DCT_img_space_loss(pred_Im, tar_Im)
        ffl = FFL()
        fflLoss=self.FFLLoss=ffl(pred_Im, tar_Im)

        # bias = 0
        bias = w1 -w2
        threshold = nn.Parameter(torch.tensor(-0.5))
        if bias <= threshold:
            bias = threshold
        if D_fake is not None:
            advLoss = self.gen_adver_loss(D_fake)
        else:
            advLoss = 0
        #DCTLoss block 3rd-Domain
        # batch_size, _, height, width = pred_Im.shape
        # pred_image_np = pred_Im.cpu().detach().numpy()
        # tar_image_np = tar_Im.cpu().detach().numpy()
        pred_dct = (dct2d(pred_Im)).clone().detach()
        tar_dct = (dct2d(tar_Im)).clone().detach()
        # pred_dct = torch.zeros_like(torch.tensor(pred_image_np))
        # tar_dct = torch.zeros_like(torch.tensor(tar_image_np))
        # for i in range(batch_size):
        #      pred_dct_1batch = dct(dct(pred_image_np[i, 0, :, :], norm='ortho').T, norm='ortho')
        #      tar_dct_1batch = dct(dct(tar_image_np[i, 0, :, :], norm='ortho').T, norm='ortho')
        #      pred_dct[i, 0, :, :] = torch.tensor(pred_dct_1batch)
        #      tar_dct[i, 0, :, :] = torch.tensor(tar_dct_1batch)
        dct_space = dct2d(pred_Im)
        DCTLoss = self.dct_space_loss(dct_space)
        # Sparse Loss + total variation penalty
        # DCTLoss = torch.tensor(0.1)
        SSIMLoss = 1 - mt.ssim(tar_Im, pred_Im)

        fullLoss = self.ImL2_weights*ImL2 + self.ImL1_weights*ImL1 + self.AdverLoss_weight*advLoss + self.KspaceL2_weights * KspaceL2 +\
            self.DCT_imgLoss_weight * DCT_imgLoss + self.DCTLoss_weight * DCTLoss + SSIMLoss + self.FFLLoss_weight*fflLoss#+ bias * 0.1
        return fullLoss, ImL2, ImL1, KspaceL2, advLoss, fflLoss, DCTLoss, DCT_imgLoss, SSIMLoss

    def calc_disc_loss(self,D_real,D_fake):
        real_loss,fake_loss = self.disc_adver_loss(D_real,D_fake)
        return real_loss,fake_loss, 1*(0.5*(real_loss + fake_loss))

def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad
# class netLoss(Function):
#     """Dice coeff for individual examples"""
#
#     def forward(self, input, target):
#         self.save_for_backward(input, target)
#         eps = 0.0001
#         self.inter = torch.dot(input.view(-1), target.view(-1))
#         self.union = torch.sum(input) + torch.sum(target) + eps
#
#         t = (2 * self.inter.float() + eps) / self.union.float()
#         return t
#
#     # This function has only a single output, so it gets only one gradient
#     def backward(self, grad_output):
#
#         input, target = self.saved_variables
#         grad_input = grad_target = None
#
#         if self.needs_input_grad[0]:
#             grad_input = grad_output * 2 * (target * self.union - self.inter) \
#                          / (self.union * self.union)
#         if self.needs_input_grad[1]:
#             grad_target = None
#
#         return grad_input, grad_target
#
#
# def dice_coeff(input, target):
#     """Dice coeff for batches"""
#     if input.is_cuda:
#         s = torch.FloatTensor(1).cuda().zero_()
#     else:
#         s = torch.FloatTensor(1).zero_()
#
#     for i, c in enumerate(zip(input, target)):
#         s = s + netLoss().forward(c[0], c[1])
#
#     return s / (i + 1)
def total_variation(image):
    # Compute gradients using finite differences
    image = image.cpu().detach().numpy()
    dx = np.diff(image, axis=-1)
    dy = np.diff(image, axis=-2)
    dx = np.pad(dx, ((0,0), (0,0), (0,0), (0,1)), mode='constant')
    dy = np.pad(dy, ((0,0), (0,0), (1,0), (0,0)), mode='constant')
    tv = np.sum(np.sqrt(dx**2 + dy**2))
    tv = torch.tensor(tv)
    return tv

def FourierTransform(image):
    # Fourier Transform
    kspace_cmplx = torch.fft.fft2(image, dim=[2,3])
    kspace_cmplx = torch.fft.fftshift(kspace_cmplx, dim=[2,3])
    kspace_real = kspace_cmplx.real
    kspace_imag = kspace_cmplx.imag
    kspace = torch.cat((kspace_real, kspace_imag), dim=1)
    return kspace

def dct2d(input_tensor):
    input_array = input_tensor.cpu().detach().numpy()
    dct_array = dctn(input_array, type=2, norm='ortho', axes=[2,3])
    # plt.imshow(abs(dct_array[0,0,:,:]), cmap='gray')
    # plt.title('calibration')
    # plt.show()
    dct_tensor = torch.tensor(dct_array)
    return dct_tensor
