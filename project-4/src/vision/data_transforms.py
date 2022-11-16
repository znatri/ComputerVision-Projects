"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fundamental_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    fundamental_transforms = transforms.Compose([ 
        transforms.Resize(inp_size),
        transforms.ToTensor()
    ])

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    fund_aug_transforms = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    fund_aug_transforms = transforms.Compose([ 
        transforms.Resize(inp_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(),
        # transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomCrop(size=inp_size),
        transforms.ToTensor(),
    ])

    ###########################################################################
    # Student code end
    ###########################################################################
    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model alomg with
    normalization.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fund_norm_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    fund_norm_transforms = transforms.Compose([ 
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std)
    ])

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set,
    along with normalization. This should just be your previous method + normalization.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    all_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    # Bad transforms
    # all_transforms = transforms.Compose([ 
    #     transforms.Resize(inp_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=pixel_mean, std=pixel_std),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.ColorJitter(),
    #     transforms.RandomRotation(degrees=(-180, 180)),
    #     transforms.RandomCrop(size=inp_size)
    # ])

    all_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(),
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std)
    ])

    ###########################################################################
    # Student code ends
    ###########################################################################
    return all_transforms
