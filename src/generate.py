# Flux Fine-tuning Script for Google Colab
# Note: Run this in a Colab notebook

# Install required packages
!pip install -U pip
!pip install torch torchvision torchaudio
!pip install -U git+https://github.com/huggingface/diffusers.git
!pip install -U git+https://github.com/huggingface/transformers.git
!pip install -U git+https://github.com/huggingface/accelerate.git
!pip install -U git+https://github.com/huggingface/peft.git
!pip install bitsandbytes
!pip install wandb

# Clone SimpleTuner repository
!git clone --branch=release https://github.com/bghira/SimpleTuner.git
%cd SimpleTuner

# Install SimpleTuner dependencies
!pip install -U poetry
!poetry install

# Set up environment variables
import os
os.environ["HF_HOME"] = "/content/hf_home"
os.environ["WANDB_PROJECT"] = "flux-finetuning"

# Login to Hugging Face and Weights & Biases
from huggingface_hub import login
login()

import wandb
wandb.login()

# Create necessary directories
!mkdir -p config datasets/my_images

# Upload your images to the 'my_images' folder in Colab
# You can do this manually through the Colab interface

# Create config files
!cp config/config.json.example config/config.json

# Modify config.json (you may need to adjust these values)
!sed -i 's/"model_type": "sd"/"model_type": "lora"/g' config/config.json
!sed -i 's/"model_family": "stable-diffusion"/"model_family": "flux"/g' config/config.json
!sed -i 's/"pretrained_model_name_or_path": "runwayml\/stable-diffusion-v1-5"/"pretrained_model_name_or_path": "black-forest-labs\/FLUX.1-dev"/g' config/config.json
!sed -i 's/"output_dir": "output"/"output_dir": "\/content\/drive\/MyDrive\/flux_output"/g' config/config.json
!sed -i 's/"train_batch_size": 4/"train_batch_size": 1/g' config/config.json
!sed -i 's/"max_train_steps": 100000/"max_train_steps": 1000/g' config/config.json
!sed -i 's/"save_steps": 500/"save_steps": 100/g' config/config.json
!sed -i 's/"mixed_precision": "fp16"/"mixed_precision": "bf16"/g' config/config.json
!sed -i 's/"base_model_precision": "default"/"base_model_precision": "int8-quanto"/g' config/config.json

# Create multidatabackend.json
cat << EOF > config/multidatabackend.json
[
  {
    "id": "my-images",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/my-images",
    "instance_data_dir": "datasets/my_images",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "a photo of myself",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux",
    "disabled": false,
    "write_batch_size": 128
  }
]
EOF

# Start training
!python train.py \
    --data_backend_config=config/multidatabackend.json \
    --config_file=config/config.json \
    --flux_lora_target=all \
    --lora_rank=4 \
    --learning_rate=1e-4 \
    --max_grad_norm=1.0 \
    --optimizer=adamw_bnb_8bit \
    --report_to=wandb
