"""Initialize CIFAR-stem ResNet18-GN from official torchvision ImageNet weights."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_imagenet_gn(num_classes=10):
    """Transfer convolutions and BN affine tensors; reinitialize classifier.

    The 7x7 ImageNet stem is center-cropped to 3x3. BN running statistics
    are intentionally unused because GroupNorm has no running statistics.
    """
    weights=ResNet18_Weights.IMAGENET1K_V1
    source=weights.get_state_dict(progress=False,check_hash=False)
    model=resnet18(num_classes=num_classes,norm_layer=lambda c:nn.GroupNorm(8,c))
    model.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()
    current=model.state_dict()
    copied=[]
    for key in current:
        if key.startswith('fc.'):
            continue
        value=source[key]
        if key=='conv1.weight':
            value=value[:,:,2:5,2:5].contiguous()
        if value.shape != current[key].shape:
            raise ValueError(f'shape mismatch: {key}: {value.shape} vs {current[key].shape}')
        current[key]=value;copied.append(key)
    model.load_state_dict(current)
    return model, dict(source='torchvision ResNet18_Weights.IMAGENET1K_V1',
                       source_url=weights.url,copied_tensors=len(copied),
                       stem='center crop 7x7 to 3x3; stride 1; no maxpool',
                       norm='GroupNorm(8), BN affine copied, BN running statistics discarded',
                       classifier=f'random {num_classes}-class head')
