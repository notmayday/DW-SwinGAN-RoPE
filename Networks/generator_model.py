""" Full assembly of the parts to form the complete network """
import pickle
from .unet_parts import *
from .vision_transformer import SwinUnet, SwinUnet_ROPE
class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):  #频域生成器初始参数 (self,6,2,True) 图像域初始参数 (self,1,1,True)
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """

        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class WNet(nn.Module):

    def __init__(self, args, masked_kspace=True):
        super(WNet, self).__init__()

        self.bilinear = args.bilinear
        self.args = args
        self.masked_kspace = masked_kspace
        self.w1 = nn.Parameter(torch.tensor(0.33))
        self.w2 =nn.Parameter(torch.tensor(0.33))
        self.w3 =nn.Parameter(torch.tensor(0.33))
        self.dynamic_threshold = nn.Parameter(torch.tensor(0.005))
        self.sparse_mask = torch.ones((args.batch_size, 2, args.img_size, args.img_size))
        # self.sparse_mask[:, :, 0:64, 0:64] = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sparse_mask = self.sparse_mask.to(device)
        mask_path = args.mask_path

        with open(mask_path, 'rb') as pickle_file:
            masks = pickle.load(pickle_file)
            self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
        self.maskNot = self.mask == 0

        if self.args.ST:
            self.kspace_Unet = SwinUnet(args, img_size=args.img_size, num_classes=args.num_classes + 1,in_chans=2, depths=[2, 2, 6, 2]).cuda()
            self.img_UNet_real = SwinUnet_ROPE(args, img_size=args.img_size, num_classes=args.num_classes, in_chans=4, depths=[2, 2, 6, 2]).cuda()
            self.img_UNet_imag = SwinUnet_ROPE(args, img_size=args.img_size, num_classes=args.num_classes, in_chans=4, depths=[2, 2, 6, 2]).cuda()


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


    def forward(self, Kspace, enh_img, sensitivity_map):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sensitivity_map = sensitivity_map.to(device)
        rec_ori_img = self.inverseFT(Kspace)
        rec_all_Kspace = self.kspace_Unet(Kspace)
        if self.masked_kspace:
            theta = 0.05
            rec_Kspace = (self.mask*Kspace[:, int(Kspace.shape[1]/2)-1:int(Kspace.shape[1]/2)+1, :, :] +theta * self.mask * rec_all_Kspace) / (1 + theta) +\
                         self.maskNot*rec_all_Kspace

            rec_mid_img = self.inverseFT(rec_Kspace)
            rec_mid_img_real = rec_mid_img[:,0,:,:]
            rec_mid_img_real = rec_mid_img_real[:, None, :, :]
            rec_mid_img_imag = rec_mid_img[:,1,:,:]
            rec_mid_img_imag = rec_mid_img_imag[:, None, :, :]
            rec_mid_img = torch.sqrt(rec_mid_img_real**2+rec_mid_img_imag**2)
            rec_mid_img = torch.tanh(rec_mid_img)
            rec_mid_img = torch.clamp(rec_mid_img, 0, 1)
            ### High frequency part
            rec_mid_img_real_h = self.high_frequency_image(rec_mid_img_real)
            rec_mid_img_real = torch.cat((rec_mid_img_real,rec_mid_img_real_h),dim=1)
            rec_mid_img_imag_h = self.high_frequency_image(rec_mid_img_imag)
            rec_mid_img_imag = torch.cat((rec_mid_img_imag, rec_mid_img_imag_h), dim=1)
            rec_mid_img_real = rec_mid_img_real.to(device)
            rec_mid_img_imag = rec_mid_img_imag.to(device)
            rec_mid_img_real = torch.cat((rec_mid_img_real, sensitivity_map), dim=1)
            rec_mid_img_imag = torch.cat((rec_mid_img_imag, sensitivity_map), dim=1)

        else:
            rec_Kspace = rec_all_Kspace
            rec_mid_img = self.fftshift(self.inverseFT(rec_Kspace))
            rec_mid_img_real = rec_mid_img[:,0,:,:]
            rec_mid_img_real = rec_mid_img_real[:, None, :, :]
            rec_mid_img_imag = rec_mid_img[:,1,:,:]
            rec_mid_img_imag = rec_mid_img_imag[:, None, :, :]
            rec_mid_img = torch.sqrt(rec_mid_img_real**2+rec_mid_img_imag**2)
            rec_mid_img = torch.tanh(rec_mid_img)
            rec_mid_img = torch.clamp(rec_mid_img, 0, 1)

            ### High frequency part
            rec_mid_img_real_h = self.high_frequency_image(rec_mid_img_real)
            rec_mid_img_real = torch.cat((rec_mid_img_real,rec_mid_img_real_h),dim=1)
            rec_mid_img_imag_h = self.high_frequency_image(rec_mid_img_imag)
            rec_mid_img_imag = torch.cat((rec_mid_img_imag, rec_mid_img_imag_h), dim=1)
            rec_mid_img_real = rec_mid_img_real.to(device)
            rec_mid_img_imag = rec_mid_img_imag.to(device)
            rec_mid_img_real = torch.cat((rec_mid_img_real, sensitivity_map), dim=1)
            rec_mid_img_imag = torch.cat((rec_mid_img_imag, sensitivity_map), dim=1)
        ## real + imaginenary Transformer

        rec_mid_img = rec_mid_img.to(device)
        refine_Img_real = self.img_UNet_real(rec_mid_img_real)
        refine_Img_imag = self.img_UNet_imag(rec_mid_img_imag)
        refine_Img=torch.sqrt(refine_Img_real**2+refine_Img_imag**2)
        refine_Img = torch.tanh(refine_Img)
        refine_Img= torch.clamp(refine_Img, 0, 1)
        rec_img = self.w1 * rec_mid_img + self.w2 * refine_Img + self.w3 *enh_img
        rec_img = torch.tanh(rec_img)
        rec_img = torch.clamp(rec_img, 0, 1)




        return rec_img, rec_Kspace, rec_mid_img, enh_img, self.w1, self.w2, self.w3
