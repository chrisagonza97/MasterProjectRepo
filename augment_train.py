#https://kornia.readthedocs.io/en/latest/augmentation.module.html
#https://kornia.readthedocs.io/en/latest/augmentation.container.html#kornia.augmentation.container.AugmentationSequential
import kornia.augmentation as K
from kornia.constants import Resample, DataKey
import torch


augment = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.2),
    K.RandomRotation(degrees=20, resample=Resample.NEAREST),
    data_keys=[DataKey.INPUT, DataKey.MASK],  # Ensures same transformation for image & mask
    same_on_batch=True
)

color_jitter = K.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5, same_on_batch=True)

def augment_train(X_train, Y_train, augment_times=2):
    Y_train = Y_train.unsqueeze(1)
    X_train_aug, Y_train_aug = [X_train], [Y_train]
    
    
    for _ in range(augment_times):
        X_aug_list, Y_aug_list = [], []

        # Process each image-mask pair individually
        for i in range(len(X_train)):
            img = X_train[i].unsqueeze(0)  # Add batch dim -> (1, C, H, W)
            mask = Y_train[i].unsqueeze(0)  # Add batch dim -> (1, 1, H, W)

            # Apply geometric transformations
            geometric = augment(img, mask)

            # Apply color jitter only to images
            augmented_x = color_jitter(geometric[0])  # First element is the transformed image
            augmented_y = geometric[1]  # Keep mask unchanged

            X_aug_list.append(augmented_x)
            Y_aug_list.append(augmented_y)

        X_train_aug.append(torch.cat(X_aug_list, dim=0))
        Y_train_aug.append(torch.cat(Y_aug_list, dim=0))

        
    # Final concatenation
    X_train_aug = torch.cat(X_train_aug, dim=0) 
    Y_train_aug = torch.cat(Y_train_aug, dim=0)

    Y_train_aug = Y_train_aug.squeeze(1)

    print(f"Augmented X shape: {X_train_aug.shape}, Augmented Y shape: {Y_train_aug.shape}")
    return X_train_aug, Y_train_aug
    
