from torchvision import transforms

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Returns the appropriate transform pipeline.

    Args:
        augment: If True, applies domain-specific augmentations
                 suited for port CCTV imagery (varying viewpoints,
                 lighting conditions, scale). If False, only applies
                 resize and normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_train_transforms() -> transforms.Compose:
    return get_transforms(augment=True)


def get_val_transforms() -> transforms.Compose:
    return get_transforms(augment=False)