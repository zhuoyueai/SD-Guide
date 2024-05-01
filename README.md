# SD-Guide

This guide requires at least 12gb vram gpu. and at least 40gb+ ram

# Training with Kohya-ss/sd-scripts

First download the [repo](https://github.com/kohya-ss/sd-scripts) and set it up in WSL. alternatively you can use [this fork](https://github.com/bmaltais/kohya_ss), which has more script and GUI and installation script stuff . Make sure the setup your WSL to be able to use nvidia gpu if you have, I forgot how I did it but I remember I needed to. You need to set up an environment, I suggest using `python -m venv venv` or use conda/miniconda `conda create -n kohyass python=3.10`. You need to install pytorch etc and the requirements.txt as stated in the repo. If you have new GPU like 4060/4070 ... then you should install the pytorch CUDA 12.1 version for huge speed up. reason for installing in WSL is also the huge speedup, I tested it's up 2x speed up compared to Windows.

## Scraping Dataset

If dataset requires to be scraped from social media/art websites, you can use [gallery-dl](https://github.com/mikf/gallery-dl) for scraping.
Facebook doesn't work for gallery-dl you can use [this script](https://github.com/zhuoyueai/facebook-image-scraper) I made.

## Dataset captioning

If your dataset is less than 20 images, you can manually caption each image by creating a txt with the same name as the image in the same directory as the image.
Captioning wise, the more descriptive the caption the better the training will be, Clip text encoder is only 75 tokens long so you need to keep your caption under 75 tokens, if you go over 75 tokens the rest of the extra tokens will be ignored.

It is still possible to use longer captions by using the `--max_token_length=225` argument in kohyass. This is useful because llava captions are pretty long. How this is done is the vectors of the tokens are aggregated so each token weights less.

### Aesthetic captioning

A method that is used is to seperate your image folders into further subfolders and label each folder based on the asethetic quality of the images. Or adjusting the number of steps in each folder, which can be done through the toml, e.g low quality images traing 10 steps per epoch, high quality 40 steps per epoch and so on. For aesthetic captioning, you can just write a script to prepend some sort of aesthetic caption to your txt caption files. Eg. "high quality", "medium quality", "low quality" etc. though this requires a large number of images for the model to learn.
 
### Training specific tokens

In the training process, the model trys to align the diffused image to the clip encoded text vectors. This makes it possible to train certain concepts into the model. For example, if I want to train the model on the concept of a person's face. I may have like 10 images of the person. and 10 images of other people. for the 10 images with the person, I prepend to the txt caption `xperson` or some random text that you think clip has not already been trained to encode to a specific vector.

The images without the person's face are called regularization images. It helps the model understand how the images will be generated like without the token and helps with overfitting.

## Dataset preparation

once you have your dataset, you need to prepare config files (create your own script to do this if you have a large dataset), folder structure, image-masking (for lora only), image upscaling, image extension

### Folder structure

Create 3 folders like 

`/mnt/g/zhuoyueai/model`
`/mnt/g/zhuoyueai/config`
`/mnt/g/zhuoyueai/log`

Create another folder

`/mnt/g/zhuoyueai/model/sample`

Create a new file prompt.txt

`/mnt/g/zhuoyueai/model/sample/prompt.txt`

Enter sample prompts in prompt.txt. Sample images are the images that are generated every n steps in the training process to view your training progress

```
zhuoyue, BMT, a soldier in green standing in front of a sign that says WELCOME TO PULAU TEKONG --n ugly, deformed, mutated, disfigured --w 832 --h 1216 --d 1 --l 7 --s 28

zhuoyue, a man wearing a big fluffy white down jacket in Italy --n ugly, deformed, mutated, disfigured --w 832 --h 1216 --d 1 --l 7 --s 28

yuai in green military uniform standing in front of a sign that says WELCOME TO PULAU TEKONG --n ugly, deformed, mutated, disfigured --w 832 --h 1216 --d 1 --l 7 --s 28

yuai smiling at the carema posing as there is a plate of food infront of him --n ugly, deformed, mutated, disfigured --w 832 --h 1216 --d 1 --l 7 --s 28

a portrait of yuai --n ugly, deformed, mutated, disfigured --w 1024 --h 1024 --d 1 --l 7 --s 28
```

### Metadata config json files

According to the instructions in the kohya-ss repo, you need to create a json file for the metadata captions, though I have a feeling it may work without I haven't tested it yet.

The following is a example to convert 1 folder of images to a json metadata, you need to put it in the toml later.

```
python3 finetune/merge_dd_tags_to_metadata.py --full_path /mnt/g/grabber/image_folder /mnt/g/zhuoyueai/config/10_amamitsuki_repeat21.json
```

Create your own script if you have a multiple folder hierarchy with different images, steps, etc..

### toml config files
Create your own script to make your toml file based on your folder structure and images, and json metadata config

this is an toml example for full finetune

```
[general]
shuffle_caption = false
enable_bucket = true

[[datasets]]
resolution = 1024
[[datasets.subsets]]
image_dir = "/mnt/g/grabber/zhuoyue/final/1010_BMT"
metadata_file = "/mnt/g/loraimages/zhuoyueai/config/1010_BMT1.json"
num_repeats = 10

[[datasets.subsets]]
image_dir = "/mnt/g/grabber/zhuoyue/final/1010_SCS"
metadata_file = "/mnt/g/loraimages/zhuoyueai/config/1010_SCS2.json"
num_repeats = 10

[[datasets.subsets]]
image_dir = "/mnt/g/grabber/zhuoyue/final/1200_yuai"
metadata_file = "/mnt/g/loraimages/zhuoyueai/config/1200_yuai3.json"
num_repeats = 200

[[datasets.subsets]]
image_dir = "/mnt/g/grabber/zhuoyue/final/1200_zhuoyue"
metadata_file = "/mnt/g/loraimages/zhuoyueai/config/1200_zhuoyue4.json"
num_repeats = 200
```



this is a toml exmample for lora with image masking

```
[general]
shuffle_caption = false
enable_bucket = true

[[datasets]]
resolution = 1024

[[datasets.subsets]]
image_dir = "/mnt/g/grabber/zhuoyue/final/1200_zhuoyue"
caption_extension = '.txt'
conditioning_data_dir = "/mnt/g/grabber/zhuoyue/final/1200_zhuoyue/conditioning"
num_repeats = 100
```


### image-masking

You can create a black and white mask for the images you are training. This helps the training focus on specific areas of the image when training. Training loss is only calculated for the white areas of the mask, so the black areas of the image mask will not be learnt at all. If it's grey then it only takes half the loss and so on.

create a folder in your image directory like `/conditioning` or something, and put all the masks there with the same name as the image. Use the ` --masked_loss` argument in the training script

Example of masking the head only for training
![image](https://github.com/zhuoyueai/SD-Guide/assets/168568061/56ffbeda-e0dd-41de-b892-d8ac1213c41e)


### image upscaling

For SD1.5 AI upscale the images to at least 512x512 area. For SDXL at least 1024x1024 area. It's not a must to upscale, you can still train with low resolution images, but generations will be worst for some reason.

### image extension

Rename jpeg extensions to jpg

Create a folder for your dataset

## Download models for training

Popular base Stable Diffusion models for finetuning:
 1. Leaked NovelAI SD 1.5 - Location ??? (Find it somewhere on huggingface) (One of the oldest but still one of the best anime models available)
 2. Stable Diffusion XL base
 3. Pony Diffusion v6 (Finetuned SDXL) (NSFW Anime model)
 4. SDXL Turbo (Much Faster than SDXL at a tiny cost of image quality)
 5. SDXL Lightning (bytedance)
 6. SDXL Hyper (bytedance)
    
## Training SDXL
I'm only gonna give example for finetuning SDXL, though SD1.5 uses less vram for training, you can switch to that if you have not enough vram, go to kohya-ss/sd-script to find out more.

### Full Finetune

You probably need at least 16gb vram for full finetune even with `--full_bf16`
Here is an example to run a full finetune (Finetune all model weights, UNET only) once you have setup your environment. 

```
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train.py --pretrained_model_name_or_path=/mnt/g/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors --output_dir=/mnt/g/zhuoyueai/model --logging_dir=/mnt/g/zhuoyueai/log --dataset_config=zhuoyueaiXL.toml --learning_rate=5e-6 --min_bucket_reso=256 --max_bucket_reso=4096 --save_model_as=safetensors --output_name=zhuoyueaiXL --lr_scheduler=constant_with_warmup --lr_warmup_steps=0 --train_batch_size=4 --gradient_checkpointing --max_train_steps=10000000 --save_every_n_epochs=1 --save_state --mixed_precision=bf16 --save_precision=bf16 --no_half_vae --cache_text_encoder_outputs_to_disk --cache_latents_to_disk --optimizer_type=Adafactor --optimizer_args="scale_parameter=False","relative_step=False","warmup_init=False" --max_data_loader_n_workers=0 --xformers --bucket_reso_steps=32 --min_snr_gamma=5 --noise_offset=0.0357 --adaptive_noise_scale=0.00357 --max_token_length=225 --bucket_no_upscale --sample_sampler=euler_a --sample_prompts=/mnt/g/loraimages/zhuoyueai/model/sample/prompt.txt --sample_every_n_steps=5000 --full_bf16
```

I remember having some error with parsing `optimizer_args` so you either remove that or manually edit the script source.
Some of the arguments you may want to change are the batch size, and the paths, `max_token_length` can be removed if you have less than 75 token. save every 1 epoch means if you have 10 images, and each images is trained 100 steps according to your config toml, it would be 1000 steps total for 1 epoch, and a new full model will be saved in the directory

### Lora

Here's an example for training a lora, lora is like a smaller network that can be stacked on top of the base model, so training this does not affect the weights of the base model.

```
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --pretrained_model_name_or_path=/mnt/g/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors --output_dir=/mnt/g/loraimages/zhuoyueai/model --logging_dir=/mnt/g/loraimages/zhuoyueai/log --dataset_config=yuzhaiaiXL_lora.toml --learning_rate=1e-6 --min_bucket_reso=256 --max_bucket_reso=4096 --save_model_as=safetensors --output_name=yuzhaiaiXL_lora_upscale --lr_scheduler=constant_with_warmup --lr_warmup_steps=0 --train_batch_size=3 --max_train_steps=10000000 --save_every_n_epochs=1 --gradient_checkpointing --cache_latents_to_disk --mixed_precision=bf16 --save_precision=bf16 --no_half_vae --optimizer_type=Adafactor --bucket_no_upscale --sample_sampler=euler_a --sample_prompts=/mnt/g/loraimages/zhuoyueai/model/sample/prompt.txt --sample_every_n_steps=400 --masked_loss --network_module=networks.lora
```

You want to increase the learning rate here, learning rates of lora are usually much higher than learning rates for full finetune.

### Not enough VRAM/Out of memory error (OOM)

- set batch-size to 1
- change optimizer to adam-8bit
- GG buy big gpu
- Google colab


## Inference with ComfyUI

https://comfyanonymous.github.io/ComfyUI_examples/

### Basic workflow

Drag and drop the image [here](https://comfyanonymous.github.io/ComfyUI_examples/sdxl/) into ComfyUI 

### Stacking Loras

https://comfyanonymous.github.io/ComfyUI_examples/lora/

# ComfyUI API

Official API example:

https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/websockets_api_example_ws_images.py

Implementation:

https://github.com/zhuoyueai/comfy-discord-bot

# Other Interesting topics

- Block weight merge
- Model merging
- ControlNet/IP Adapter
- Training Negative embeddings (reinforcement learning)

