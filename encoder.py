import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from positional import add_timing_signal_nd


class Encoder(nn.Module):
    """Class that applies convolutions to an image"""

    def __init__(self, config):
        super(Encoder, self).__init__()
        self._config = config

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        # Optional convolution for "cnn" configuration
        self.strided_conv = nn.Conv2d(512, 512, kernel_size=(2, 4), stride=2, padding=1)
    def forward(self, img, dropout=0.0):
        """Applies convolutions to the image

        Args:
            img: batch of img, shape = (B, C, H, W), of type torch.float32
            dropout: dropout rate (not used in this implementation)

        Returns:
            the encoded images, shape = (B, C', H', W')
        """
        # Normalize input to [0, 1]
        img = img.permute(0, 3, 1, 2)
        img = img / 255.0

        # First conv + max pool
        out = F.relu(self.conv1(img))
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)

        # Second conv + max pool
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)

        # Regular convs
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))

        # Optional max pooling based on configuration
        if self._config.encoder_cnn == "vanilla":
            out = F.max_pool2d(out, kernel_size=(2, 1), stride=(2, 1), padding=0)

        out = F.relu(self.conv5(out))

        # Optional max pooling based on configuration
        if self._config.encoder_cnn == "vanilla":
            out = F.max_pool2d(out, kernel_size=(1, 2), stride=(1, 2), padding=0)

        # Optional strided convolution
        if self._config.encoder_cnn == "cnn":
            out = F.relu(self.strided_conv(out))

        # Final convolution
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))

        # Add positional embeddings if enabled
        if self._config.positional_embeddings:
            out = add_timing_signal_nd(out)

        return out



    # max_shape = get_max_shape(images)
    # np.ones([len(images)] + list(images))