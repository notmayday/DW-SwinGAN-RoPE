import numpy as np
import matplotlib.pyplot as plt
from utils.grappa import grappa

def prefilling(kspaces, kernel_size):
    batch_size = kspaces.shape[0]
    img_size = kspaces.shape[2]
    filled_kspace = np.zeros(kspaces.shape)
    upper_bound = 113
    lower_bound = 142
    calib_1b = kspaces[:,upper_bound:lower_bound,upper_bound:lower_bound]
    kspaces = np.moveaxis(kspaces, 0, -1)
    calib_1b = np.moveaxis(calib_1b, 0, -1)
    kspace_filled = grappa(kspaces, calib_1b, kernel_size)
    filled_kspace =kspace_filled

    return (filled_kspace)

# import numpy as np
# from multiprocessing import Pool
# from functools import partial
# from utils.grappa import grappa
#
# def process_single_sample(kspace, kernel_size, upper_bound, lower_bound):
#     kspace = np.moveaxis(kspace, 0, -1)
#     calib = kspace[upper_bound:lower_bound, upper_bound:lower_bound, :]
#     kspace_filled = grappa(kspace, calib, kernel_size)
#     return np.moveaxis(kspace_filled, -1, 0)
#
# def prefilling(kspaces):
#     batch_size = kspaces.shape[0]
#     img_size = kspaces.shape[2]
#     kernel_size = [5, 5]
#     filled_kspace = np.zeros(kspaces.shape)
#
#     upper_bound = 113
#     lower_bound = 142
#
#     # Define the function for parallel processing
#     process_func = partial(process_single_sample, kernel_size=kernel_size,
#                            upper_bound=upper_bound, lower_bound=lower_bound)
#
#     # Use multiprocessing Pool to process samples in parallel
#     with Pool() as pool:
#         results = pool.map(process_func, kspaces)
#
#     # Fill the result into the filled_kspace array
#     for i, result in enumerate(results):
#         filled_kspace[i, :, :, :] = result
#
#     return filled_kspace