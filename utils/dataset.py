from os.path import splitext
from os import listdir,path
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import h5py
import pickle
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}



class fastMRIdataset(Dataset):
    def __init__(self, data_dir, args, validtion_flag=False):
        self.args = args
        self.data_dir = data_dir
        self.validtion_flag = validtion_flag

        self.num_input_slices = args.num_input_slices
        self.img_size = args.img_size
        #make an image id's list
        self.file_names = [splitext(file)[0] for file in listdir(data_dir)
                    if not file.startswith('.')]

        self.ids = list()
        self.calculator = 0
        for file_name in self.file_names:
            try:
                full_file_path = path.join(self.data_dir,file_name+'.hdf5')
                with h5py.File(full_file_path, 'r') as f:
                    numOfSlice = f['cropped_kspaces'].shape[0]
                    numOfCoil = f['cropped_kspaces'].shape[1]
                if numOfSlice < self.args.slice_range[1]:
                    continue

                for slice in range(self.args.slice_range[0], self.args.slice_range[1]):
                    for coil in range(0,numOfCoil):
                        self.ids.append((file_name, slice, coil))
            except:
                continue

        if self.validtion_flag:
            logging.info(f'Creating validation dataset with {len(self.ids)} examples')
        else:
            logging.info(f'Creating training dataset with {len(self.ids)} examples')

        mask_path = args.mask_path

        with open(mask_path, 'rb') as pickle_file:
            masks_dictionary = pickle.load(pickle_file)
            self.masks = masks_dictionary['mask1']
            self.maskedNot = 1 - masks_dictionary['mask1']


        #random noise:
        self.minmax_noise_val = args.minmax_noise_val

    def __len__(self):
        return len(self.ids)

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size)/2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]
    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    # @classmethod
    def slice_preprocess(self, kspace_cplx, kspace_ori, slice_num):
        #crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx)
        #split to real and imaginary channels
        kspace = np.zeros((self.img_size, self.img_size, 2))
        kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
        #target image:
        image = self.ifft2(kspace_cplx)
        # HWC to CHW
        kspace = kspace.transpose((2, 0, 1))
        masked_Kspace = kspace*self.masks
        masked_Kspace_cplx = (kspace_ori) * self.masks
        undersampled_img = np.fft.fftshift(self.ifft2((masked_Kspace_cplx)))
        return masked_Kspace, kspace, image, undersampled_img

    def __getitem__(self, i):
        i = self.calculator
        file_name, slice_num, coil_num = self.ids[i]
        self.calculator = self.calculator + 1
        full_file_path = path.join(self.data_dir,file_name + '.hdf5')

        with h5py.File(full_file_path, 'r') as f:
            add = int(self.num_input_slices / 2)
            kspaces = f['cropped_kspaces'][slice_num-add:slice_num+add+1, coil_num, :, :, ]
            imgs = f['cropped_imgs'][slice_num-add:slice_num+add+1, coil_num, :, :, ]
            rss_imgs = f['rss_imgs'][slice_num-add:slice_num+add+1, :, :, ]
            kspaces_multicoil = f['cropped_kspaces'][slice_num-add:slice_num+add+1, : , :, :, ]
            kspaces_multicoil = kspaces_multicoil * self.masks[np.newaxis,np.newaxis,:,:]
            undersampled_imgs = abs(np.fft.ifftshift(np.fft.ifft2(kspaces_multicoil),axes=[2,3]))
            rss_img_undersampled = np.sqrt(np.sum(np.abs(undersampled_imgs) ** 2, axis=1))
            kspaces = np.transpose(kspaces, (1, 2, 0))
            imgs = np.transpose(imgs, (1, 2, 0))

        masked_Kspaces = np.zeros((self.num_input_slices*2, self.img_size, self.img_size))
        target_Kspace = np.zeros((2, self.img_size, self.img_size))
        target_img = np.zeros((1, self.img_size, self.img_size))
        sensitivity_map = np.zeros((1, self.img_size, self.img_size))

        for sliceNum in range(self.num_input_slices):
            img = abs(imgs)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.squeeze(img)
            kspace = self.fft2(img)
            kspace_ori = kspaces[:, :, sliceNum]
            slice_masked_Kspace, slice_full_Kspace, slice_full_img, slice_undersampled_img = self.slice_preprocess(kspace, kspace_ori, sliceNum)
            masked_Kspaces[sliceNum*2:sliceNum*2+2, :, :] = slice_masked_Kspace
            img_undersampled = slice_undersampled_img
            sensitivity_map = img_undersampled/rss_img_undersampled
            if sliceNum == int(self.num_input_slices/2):
                target_Kspace = slice_full_Kspace
                target_img = slice_full_img
                ori_Kspace = slice_masked_Kspace

        return {'masked_Kspaces': torch.from_numpy(masked_Kspaces), 'target_Kspace': torch.from_numpy(target_Kspace),
                'target_img': torch.from_numpy(target_img),'ori_Kspace': torch.from_numpy(ori_Kspace),'sensitivity_map': torch.from_numpy(sensitivity_map)}

