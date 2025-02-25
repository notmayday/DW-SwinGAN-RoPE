""" Full assembly of the parts to form the complete network """
import pickle
from .unet_parts import *
from .vision_transformer import SwinUnet, SwinUnet_ROPE

class Generator_PRE(nn.Module):

    def __init__(self, args, masked_kspace=True):
        super(Generator_PRE, self).__init__()

        self.bilinear = args.bilinear
        self.args = args
        self.masked_kspace = masked_kspace
        self.dynamic_threshold = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        mask_path = args.mask_path


        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
            self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
        self.maskNot = self.mask == 0

        if self.args.ST:
            self.kspace_Unet = SwinUnet(args, img_size=args.img_size, num_classes=args.num_classes + 1, in_chans=2,
                                        depths=[2, 2, 6, 2]).cuda()
            self.img_UNet_real = SwinUnet_ROPE(args, img_size=args.img_size, num_classes=args.num_classes, in_chans=4,
                                  depths=[2, 2, 6, 2]).cuda()
            self.img_UNet_imag = SwinUnet_ROPE(args, img_size=args.img_size, num_classes=args.num_classes, in_chans=4,
                                  depths=[2, 2, 6, 2]).cuda()

    def fftshift(self, img):

        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)
        K = Kspace[:,:,:,0] + 1j * Kspace[:,:,:,1]
        img_cmplx = torch.fft.ifft2(K, dim=[1,2])
        img = torch.zeros(img_cmplx.size(0), 2, img_cmplx.size(1), img_cmplx.size(2))
        img[:, 0, :, :] = img_cmplx.real
        img[:, 1, :, :] = img_cmplx.imag
        return img

    def FT(self, imgs):
        imgs = imgs.permute(0, 2, 3, 1)
        I = imgs[:,:,:,0] + 1j * imgs[:,:,:,1]
        kspace_cmplx = torch.fft.fft2(I, dim=[1,2])
        kspace = torch.zeros(kspace_cmplx .size(0), 2, kspace_cmplx .size(1), kspace_cmplx .size(2))
        kspace[:, 0, :, :] = kspace_cmplx.real
        kspace[:, 1, :, :] = kspace_cmplx.imag
        return kspace

    def high_frequency_image(self, images_tensor):
        # Convert the tensor to float and move it to device if available
        images_tensor = images_tensor.float()


        laplacian_filter = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float).view(1, 1, 3, 3)
        laplacian_filter = laplacian_filter.to(laplacian_filter.device)

        scharr_filter = torch.tensor([[[[-3, 0, 3],
                                        [-10, 0, 10],
                                        [-3, 0, 3]]]], dtype=torch.float).view(1, 1, 3, 3)
        scharr_filter = scharr_filter.to(scharr_filter.device)

        scharr_filter_T = torch.tensor([[[[-3, -10, -3],
                                        [0, 0, 0],
                                        [3, 10, 3]]]], dtype=torch.float).view(1, 1, 3, 3)
        scharr_filter_T = scharr_filter_T.to(scharr_filter_T.device)


        high_freq_images_ST = F.conv2d(images_tensor, scharr_filter_T , padding=1)
        high_freq_images_S = F.conv2d(images_tensor, scharr_filter, padding=1)
        high_freq_images = torch.cat((high_freq_images_S, high_freq_images_ST), dim=1)


        return high_freq_images



    def forward(self, Kspace, sensitivity_map):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sensitivity_map = sensitivity_map.to(device)
        rec_all_Kspace = self.kspace_Unet(Kspace)
        theta = 0.05
        rec_Kspace = (self.mask * Kspace[:, int(Kspace.shape[1] / 2) - 1:int(Kspace.shape[1] / 2) + 1, :,
                                  :] + theta * self.mask * rec_all_Kspace) / (1 + theta) + \
                     self.maskNot * rec_all_Kspace
        rec_ori_img = self.inverseFT(rec_Kspace)

        ori_img_real = rec_ori_img[:, 0, :, :]
        ori_img_real = ori_img_real[:, None, :, :]
        ori_img_imag = rec_ori_img[:, 1, :, :]
        ori_img_imag = ori_img_imag[:, None, :, :]
        ori_img = torch.sqrt(ori_img_real ** 2 + ori_img_imag ** 2)
        ori_img = torch.tanh(ori_img)
        ori_img = torch.clamp(ori_img, 0, 1)
        ### High frequency part
        ori_img_real_h = self.high_frequency_image(ori_img_real)
        ori_img_real = torch.cat((ori_img_real, ori_img_real_h), dim=1)
        ori_img_imag_h = self.high_frequency_image(ori_img_imag)
        ori_img_imag = torch.cat((ori_img_imag, ori_img_imag_h), dim=1)
        ori_img_real = ori_img_real.to(device)
        ori_img_imag = ori_img_imag.to(device)
        ori_img_real = torch.cat((ori_img_real, sensitivity_map), dim=1)
        ori_img_imag = torch.cat((ori_img_imag, sensitivity_map), dim=1)
        dynamic_threshold = self.dynamic_threshold
        cs_img_real = self.img_UNet_real(ori_img_real)
        cs_img_imag = self.img_UNet_imag(ori_img_imag)
        cs_img = torch.cat((cs_img_real,cs_img_imag), dim=1)
        enh_mid_img = cs_img
        enh_Kspace = self.FT(enh_mid_img)
        enh_Kspace = enh_Kspace.to(device)
        enh_Kspace = (self.mask*Kspace[:, int(Kspace.shape[1]/2)-1:int(Kspace.shape[1]/2)+1, :, :] +theta * self.mask * enh_Kspace) / (1 + theta) +\
                         self.maskNot*enh_Kspace
        enh_mid_img = self.inverseFT(enh_Kspace)
        enh_mid_img = enh_mid_img.to(device)
        enh_mid_img_real = enh_mid_img[:, 0, :, :]
        enh_mid_img_real = enh_mid_img_real[:, None, :, :]
        enh_mid_img_imag = enh_mid_img[:, 1, :, :]
        enh_mid_img_imag = enh_mid_img_imag[:, None, :, :]
        enh_mid_img = torch.sqrt(enh_mid_img_real ** 2 + enh_mid_img_imag ** 2)
        enh_img = torch.tanh(enh_mid_img)
        enh_img = torch.clamp(enh_img, 0, 1)


        return enh_Kspace, enh_img, ori_img, dynamic_threshold
