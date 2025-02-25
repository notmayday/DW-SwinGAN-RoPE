# DW-SwinGAN-RoPE for prostate MRI reconstruction

Official PyTorch implementation of DW-SwinGAN-RoPE for prostate MRI reconstruction described in the paper "DW-SwinGAN-RoPE: Advancing MRI Dual-Domain Reconstruction with Dynamic Weighting and Rotary Position Embedding Attention".
<br />
<br />
*Rotary Position Embedding Attention of Swin-transformer
<br />
<br />
*Dynamic weights of frequency domain and image domain

<br />
<br />

<div align="center">
  <figure> 
    <img src="./asserts/framework_block.png" width="800px">
    <figcaption><br />Pre-processing Block for Multi-coil MRI Raw Data</figcaption> 
  </figure> 
</div>

<br />
<br />

<div align="center">
  <figure> 
    <img src="./asserts/RoPE Attention.png" width="800px">
    <figcaption><br />Proposed Compressed Sensing Block</figcaption> 
    <figure> 
</div>

<br />
<br />

## Dependencies

```
easydict==1.10
einops==0.8.0
focal-frequency-loss==0.3.0
h5py==3.9.0
ipython==8.14.0
matplotlib==3.7.2
nibabel==5.1.0
numpy==1.25.2
opencv-python==4.8.0.74
Pillow==10.0.0
PyWavelets==1.4.1
PyYAML==6.0.1
scikit-image==0.21.0
scipy==1.11.1
# Editable install with no version control (setuptools==68.0.0)
timm==0.9.2
torch==2.0.1
tqdm==4.65.0
```

## Installation
- Clone this repo:
```bash
git clone https://github.com/notmayday/CS-SwinGAN_MC_Rec
cd CS-SwinGAN_MC_Rec
```

## Train

<br />

```
python3 train.py 

```

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.


<br />

# Acknowledgements

This code uses libraries from [FastMRI Prostate](https://github.com/cai2r/fastMRI_prostate), [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://huggingface.co/docs/transformers/model_doc/roformer), [Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks](https://github.com/ItamarDavid/Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks) and [SwinGAN](https://github.com/learnerzx/SwinGAN) repositories.

