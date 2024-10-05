# Flux Fine-Tuning

Fine-tune Flux model on personal images easily.
Add this block to your Google Colab cell and just run it an follow the instructions:

# Steps to follow

- First login in your Google Drive and create at the root a directory called "datasets".
- Put in it all images you want to use for fine-tuning.
- Add this cell and run it:

```
git clone https://github.com/rayanramoul/flux-finetune
%cd flux-finetune
```

- Modify config/config.json field "--validation_prompt" with the prompt you want to associate to your images.
- Run the following cell:

```
!bash src/initial_setup.sh
```

# Aknowledgements

This repository is a more straightforward version of the original [SimpleTuner Flux resources](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md)
