# Flux Fine-tuning Script for Google Colab
# Note: Run this in a Colab notebook

# Install required packages
pip install -U pip
pip install torch torchvision torchaudio
pip install -U git+https://github.com/huggingface/diffusers.git
pip install -U git+https://github.com/huggingface/transformers.git
pip install -U git+https://github.com/huggingface/accelerate.git
pip install -U git+https://github.com/huggingface/peft.git
pip install bitsandbytes
pip install wandb

# Clone SimpleTuner repository
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cp config/* SimpleTuner/config/
cd SimpleTuner

# Install SimpleTuner dependencies
pip install -U poetry
poetry install

wandb login
huggingface-cli login
