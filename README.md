MA23C024_ASS2_DL
This project involves building and evaluating deep learning models on the iNaturalist dataset. It is divided into two parts:

---

### Part A: Build and Train a Custom CNN from Scratch

You will experiment with a configurable CNN model that supports the following parameters:

- **Activation Functions**: `relu`, `gelu`, `silu`, `mish`
- **Filter Scaling**: `fixed`, `double`, `half`
- **Dropout Rates**: `0.2`, `0.3`
- **Batch Normalization**: `true` / `false`
- **Data Augmentation**: `true` / `false`

**Hyperparameter Tuning**:
Use Weights & Biases (wandb) for hyperparameter sweeps:
```bash
wandb sweep sweeps/sweep_config.yaml
wandb agent your_username/sweep_id
```

**Evaluating on Test Set**:
After training, apply the best model to the test set:
```bash
python train.py --eval --checkpoint ./outputs/best_model.pth
```

The model's predictions are visualized using the `visualize_predictions` function, which shows images from the test set with their predicted and true labels. Make sure `class_names` aligns with the training set by checking:
```python
print(train_ds.class_to_idx)
print(test_dataset.class_to_idx)
```
Set up the test set and loader with:
```python
test_dataset = datasets.ImageFolder('D:/nature_12K/inaturalist_12K/test', transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=best_config['batch_size'], shuffle=False)
class_names = test_dataset.classes
```

---

### Part B: Fine-Tune a Pretrained Model

🔧 **Training with Freezing Strategies**:
```bash
cd ../partB
python fine_tune.py --strategy freeze_last --epochs 10
```

**Available strategies**:
- `freeze_all`: Freeze all pretrained layers
- `freeze_last`: Unfreeze only the last conv block
- `unfreeze_all`: Unfreeze entire model

The script automatically evaluates the model on the test dataset and logs the following:

- Accuracy
- Confusion Matrix
- Class-wise Precision and Recall

---

### Notes
- Dataset is split with 20% stratified validation.
- All code uses `torchvision.transforms` for preprocessing.
- The test set is **never** used during training or hyperparameter tuning — only for final evaluation.

## Notes
Dataset is split with 20% stratified validation.

All code uses torchvision.transforms for preprocessing.

Trained only on training/validation set — test set is untouched during tuning

