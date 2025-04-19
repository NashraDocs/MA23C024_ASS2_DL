# MA23C024_ASS2_DL

This project involves building and evaluating deep learning models on the iNaturalist dataset. It is divided into two parts:

Part A: Build and train a custom CNN from scratch.

Part B: Fine-tune a pretrained model
Custom CNN from Scratch
activation: relu, gelu, silu, mish

--filters: "fixed", "double", "half"

--dropout: 0.2, 0.3

--batchnorm: true/false

--augment: true/false
wandb sweep sweeps/sweep_config.yaml
wandb agent your_username/sweep_id

# Evaluating on Test Set
After training, to evaluate the best model:
python train.py --eval --checkpoint ./outputs/best_model.pth


# Part B: Fine-Tuning Pretrained Model
🔧 Training with Freezing Strategies
cd ../partB
python fine_tune.py --strategy freeze_last --epochs 10

# Available strategies:

freeze_all: Freeze all pretrained layers

freeze_last: Unfreeze only the last conv block

unfreeze_all: Unfreeze entire mode
The script automatically evaluates the model on test data and logs:

# Accuracy

Confusion matrix

Class-wise precision/recall

## Notes
Dataset is split with 20% stratified validation.

All code uses torchvision.transforms for preprocessing.

Trained only on training/validation set — test set is untouched during tuning

