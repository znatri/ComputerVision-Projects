#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
from email.mime import image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vision.part1 import create_Gaussian_kernel_2D


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_frequency: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff frequency.

        PyTorch requires the kernel to be of a particular shape in order to
        apply it to an image. Specifically, the kernel needs to be of shape
        (c, 1, k, k) where c is the # channels in the image. Start by getting a
        2D Gaussian kernel using your implementation from Part 1, which will be
        of shape (k, k). Then, let's say you have an RGB image, you will need to
        turn this into a Tensor of shape (3, 1, k, k) by stacking the Gaussian
        kernel 3 times.

        Args
            cutoff_frequency: int specifying cutoff_frequency 
        Returns
            kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel_2D() function from part1.py in
          this function.
        - Since the # channels may differ across each image in the dataset,
          make sure you don't hardcode the dimensions you reshape the kernel
          to. There is a variable defined in this class to give you channel
          information.
        - You can use np.reshape() to change the dimensions of a numpy array.
        - You can use np.tile() to repeat a numpy array along specified axes.
        - You can use torch.Tensor() to convert numpy arrays to torch Tensors.
        """

        ############################
        ### TODO: YOUR CODE HERE ###

        kernel = create_Gaussian_kernel_2D(cutoff_frequency)
        k = kernel.shape[0]

        kernel = np.reshape(kernel, (1, 1, k, k))
        kernel = np.tile(kernel, (self.n_channels, 1, 1, 1))
        kernel = torch.Tensor(kernel)

        ### END OF STUDENT CODE ####
        ############################

        return kernel

    def low_pass(self, x: torch.Tensor, kernel: torch.Tensor):
        """
        Applies low pass filter to the input image.

        Args:
            x: Tensor of shape (b, c, m, n) where b is batch size
            kernel: low pass filter to be applied to the image
        Returns:
            filtered_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
          convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
          convolution function. This represents the # of channels that the
          filter will be applied to.
        """

        ############################
        ### TODO: YOUR CODE HERE ###

        padding = (kernel.shape[3]//2, kernel.shape[2]//2)
        filtered_image = F.conv2d(x, kernel, padding = padding, groups = self.n_channels)

        ### END OF STUDENT CODE ####
        ############################

        return filtered_image

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, cutoff_frequency: torch.Tensor
    ):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the
        hybrid image.

        Args:
            image1: Tensor of shape (b, c, m, n)
            image2: Tensor of shape (b, c, m, n)
            cutoff_frequency: Tensor of shape (b)
        Returns:
            low_frequencies: Tensor of shape (b, c, m, n)
            high_frequencies: Tensor of shape (b, c, m, n)
            hybrid_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function
          in this function.
        - Similar to Part 1, you can get just the high frequency content of an
          image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You
          can use torch.clamp().
        - If you want to use images with different dimensions, you should
          resize them in the HybridImageDataset class using
          torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        ############################
        ### TODO: YOUR CODE HERE ###

        kernel = self.get_kernel(cutoff_frequency)
        low_frequencies = self.low_pass(image1, kernel)
        high_frequencies = image2 - self.low_pass(image2, kernel)
        img = low_frequencies + high_frequencies
        hybrid_image = torch.clamp(img, 0, 1)

        ### END OF STUDENT CODE ####
        ############################

        return low_frequencies, high_frequencies, hybrid_image
