from torchvision import transforms as T

# Mean and standard deviation used for ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UnNormalize(T.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def transform_with_norm(mean, std):
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def inverse_transform_with_norm(mean, std):
    return T.Compose([UnNormalize(mean=mean, std=std), T.ToPILImage()])
