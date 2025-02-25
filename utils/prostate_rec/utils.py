import numpy as np

from numpy.fft import fftshift, ifftshift, ifftn
from typing import List, Optional, Sequence, Tuple
from utils.prostate_rec.grappa import Grappa
from utils.prostate_rec.mri_data import zero_pad_kspace_hdr

def ifftnd(kspace: np.ndarray, axes: Optional[Sequence[int]] = [-1]) -> np.ndarray:
    """
    Compute the n-dimensional inverse Fourier transform of the k-space data along the specified axes.

    Parameters:
    -----------
    kspace: np.ndarray
        The input k-space data.
    axes: list or tuple, optional
        The list of axes along which to compute the inverse Fourier transform. Default is [-1].

    Returns:
    --------
    img: ndarray
        The output image after inverse Fourier transform.
    """

    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)   
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))    

    return img


def flip_im(vol, slice_axis):
    """
    Flips a 3D image volume along the slice axis.

    Parameters
    ----------
    vol : numpy.ndarray of shape (slices, height, width)
        The 3D image volume to be flipped.
    slice_axis : int
        The slice axis along which to perform the flip

    Returns
    -------
    numpy.ndarray
        The flipped 3D image volume 
    """

    for i in range(vol.shape[slice_axis]):
        vol[i] = np.flipud(vol[i])
    return vol

  
def center_crop_im(im_3d: np.ndarray, crop_to_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop an image to a given size.
    
    Parameters:
    -----------
    im_3d : numpy.ndarray
        Input image of shape (slices, x, y).
    crop_to_size : list
        List containing the target size for x and y dimensions.
    
    Returns:
    --------
    numpy.ndarray
        Center cropped image of size {slices, x_cropped, y_cropped}. 
    """
    x_crop = im_3d.shape[2]/2 - crop_to_size[0]/2
    y_crop = im_3d.shape[1]/2 - crop_to_size[1]/2

    return im_3d[:, int(y_crop):int(crop_to_size[1] + y_crop), int(x_crop):int(crop_to_size[0] + x_crop)]


def t2_reconstruction(kspace_data: np.ndarray, calib_data: np.ndarray, hdr: str) -> np.ndarray:
    """
    Perform T2-weighted image reconstruction using GRAPPA technique.

    Parameters:
    -----------
    kspace_data: numpy.ndarray
        Input k-space data with shape (num_aves, num_slices, num_coils, num_ro, num_pe)
    calib_data: numpy.ndarray
        Calibration data for GRAPPA with shape (num_slices, num_coils, num_pe_cal)
    hdr: str
         The XML header string.

    Returns:
    --------
    im_final: numpy.ndarray
        Reconstructed image with shape (num_slices, 320, 320)
    """
    num_avg, num_slices, num_coils, num_ro, num_pe = kspace_data.shape

    # Calib_data shape: num_slices, num_coils, num_pe_cal
    grappa_weight_dict = {}
    grappa_weight_dict_2 = {}

    kspace_slice_regridded = kspace_data[0, 0, ...]
    grappa_obj = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)

    kspace_slice_regridded_2 = kspace_data[1, 0, ...]
    grappa_obj_2 = Grappa(np.transpose(kspace_slice_regridded_2, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)

    # calculate GRAPPA weights
    for slice_num in range(num_slices):
        calibration_regridded = calib_data[slice_num, ...]
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )
        grappa_weight_dict_2[slice_num] = grappa_obj_2.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )

    # apply GRAPPA weights
    kspace_post_grappa_all = np.zeros(shape=kspace_data.shape, dtype=complex)

    for average, grappa_obj, grappa_weight_dict in zip(
            [0, 1, 2],
            [grappa_obj, grappa_obj_2, grappa_obj],
            [grappa_weight_dict, grappa_weight_dict_2, grappa_weight_dict]
    ):
        for slice_num in range(num_slices):
            kspace_slice_regridded = kspace_data[average, slice_num, ...]
            kspace_post_grappa = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weight_dict[slice_num]
            )
            kspace_post_grappa_all[average, slice_num, ...] = np.moveaxis(np.moveaxis(kspace_post_grappa, 0, 1), 1, 2)

    # recon image for each average
    im = np.zeros((num_avg, num_slices, num_ro, num_ro))
    kspace = np.zeros((num_avg, num_slices, num_coils, num_ro, num_ro), dtype=np.complex128)
    for average in range(num_avg):
        kspace_grappa = kspace_post_grappa_all[average, ...]
        kspace_grappa_padded = zero_pad_kspace_hdr(hdr, kspace_grappa)
        kspace[average] = kspace_grappa_padded
    #     im[average] = create_coil_combined_im(kspace_grappa_padded)
    #
    # im_3d = np.mean(im, axis=0)
    # # center crop image to 320 x 320
    # img_dict = {}
    # img_dict['reconstruction_rss'] = center_crop_im(im_3d, [320, 320])


    return kspace


def create_coil_combined_im(multicoil_multislice_kspace: np.ndarray) -> np.ndarray:
    """
    Create a coil combined image from a multicoil-multislice k-space array.

    Parameters:
    -----------
    multicoil_multislice_kspace : array-like
        Input k-space data with shape (slices, coils, readout, phase encode).

    Returns:
    --------
    image_mat : array-like
        Coil combined image data with shape (slices, x, y).
    """

    k = multicoil_multislice_kspace
    image_mat = np.zeros((k.shape[0], k.shape[2], k.shape[3]))
    for i in range(image_mat.shape[0]):
        data_sl = k[i, :, :, :]
        image = ifftnd(data_sl, [1, 2])
        image = rss(image, axis=0)
        image_mat[i, :, :] = np.flipud(image)
    return image_mat

def rss(sig: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the Root Sum-of-Squares (RSS) value of a complex signal along a specified axis.

    Parameters
    ----------
    sig : np.ndarray
        The complex signal to compute the RMS value of.
    axis : int, optional
        The axis along which to compute the RMS value. Default is -1.

    Returns
    -------
    rss : np.ndarray
        The RSS value of the complex signal along the specified axis.
    """
    return np.sqrt(np.sum(abs(sig)**2, axis))