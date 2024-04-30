# SD-Guide

This guide requires at least 12gb vram gpu.

# Training with Kohya-ss/sd-scripts

First download the [repo](https://github.com/kohya-ss/sd-scripts) and set it up in WSL. Make sure the setup your WSL to be able to use nvidia gpu if you have, I forgot how I did it but I remember I needed to. You need to set up an environment, I suggest using `python -m venv venv` or use conda/miniconda `conda create -n kohyass python=3.10`. You need to install pytorch and the requirements.txt as stated in the repo. If you have new GPU like 4060/4070 ... then you should install the pytorch CUDA 12.1 version for huge speed up. reason for installing in WSL is also the huge speedup, I tested it's up 2x speed up compared to Windows.

## Scraping Dataset

If dataset requires to be scraped from social media/art websites, you can use [gallery-dl](https://github.com/mikf/gallery-dl) for scraping.
Facebook doesn't work for gallery-dl you can use [this script](https://github.com/zhuoyueai/facebook-image-scraper) I made.

## Dataset preparation

once you have your dataset, you need to prepare config files (create your own script to do this if you have a large dataset), folder structure, image-masking (for lora only), image upscaling, image extension

### Folder structure


### config files


### image-masking

There is a 

### image upscaling

For SD1.5 AI upscale the images to at least 512x512 area. For SDXL at least 1024x1024 area. It's not a must to upscale, you can still train with low resolution images, but generations will be worst for some reason.

### image extension

Rename jpeg extensions to jpg

Create a folder for your dataset

## Download models for training

Popular base Stable Diffusion models for finetuning:
 1. Leaked NovelAI SD 1.5 - Location ??? (Find it somewhere on huggingface)
 2. Stable Diffusion XL base
 3. Pony Diffusion v6 (Finetuned SDXL)
 4. Stable Diffusion XL Turbo (Faster than SDXL at a tiny cost of image quality, I haven't tested it yet)
    
## Training SDXL
SDXL is superior to SD1.5 so I'm only gonna give example for finetuning SDXL

## Inference with ComfyUI

## Basic workflow

## Stacking Loras

## Block Weight Merge

## 


