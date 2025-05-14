###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
import cv2
import numpy as np
import scipy.ndimage

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        #self.mask_ratio = None

    def forward(self, input, mask_in=None, cookie_cutter=False):
        assert len(input.shape) == 4

        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                if cookie_cutter:
                    original_mask = mask.clone()
                    mask = self.expand_white_pixels(mask.squeeze(), expansion=1)
                    mask = mask.unsqueeze(0).unsqueeze(0)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                # self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                # self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        to_forward = torch.mul(input, mask) if mask_in is not None else input
        output = super(PartialConv2d, self).forward(to_forward)

        ### check - 2 cookie cutter
        if cookie_cutter and mask_in is not None:
            self.update_mask = F.conv2d(original_mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                        padding=self.padding, dilation=self.dilation, groups=1)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)

        # if self.bias is not None:
        #    bias_view = self.bias.view(1, self.out_channels, 1, 1)
        #    output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
        #    output = torch.mul(output, self.update_mask)
        # else:
        #    output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

   #def expand_white_pixels(self, mask, expansion=1):
   #    if isinstance(mask, torch.Tensor):
   #        mask = mask.cpu().numpy()

   #    mask_uint8 = (mask * 255).astype(np.uint8)
   #    kernel = np.ones((3, 3), np.uint8)

   #    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=expansion)
   #    dilated_mask_binary = (dilated_mask / 255).astype(np.float32)
   #    dilated_mask_tensor = torch.from_numpy(dilated_mask_binary)

   #    return dilated_mask_tensor

    def expand_white_pixels(self, mask, expansion=1):
       # Ensure the mask is squeezed to remove any singleton dimensions
       if isinstance(mask, torch.Tensor):
           original_device = mask.device  # Store the device information
           mask = mask.squeeze()  # Remove all singleton dimensions

           # If the mask is still 3D, it likely has multiple channels
           if mask.dim() == 3:
               # Option 1: Select the first channel (if appropriate)
               mask = mask[0, :, :]
               # Option 2: Average across channels to get a single 2D mask
               # mask = mask.mean(dim=0)

           if mask.dim() > 2:
               raise ValueError(f"Mask should be 2D, but got {mask.dim()}D after squeezing.")

           # Convert the mask to a 2D numpy array
           mask = mask.cpu().numpy()

       # Now mask should be a 2D numpy array
       mask_uint8 = (mask * 255).astype(np.uint8)
       kernel = np.ones((3, 3), np.uint8)
       dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=expansion)
       dilated_mask_binary = (dilated_mask / 255).astype(np.float32)

       # Convert back to a PyTorch tensor and move to the original device
       dilated_mask_tensor = torch.from_numpy(dilated_mask_binary).to(original_device)

       return dilated_mask_tensor

    def shrink_white_pixels(self, mask, erosion=1):
        # Ensure the mask is squeezed to remove any singleton dimensions
        if isinstance(mask, torch.Tensor):
            original_device = mask.device  # Store the device information
            mask = mask.squeeze()  # Remove all singleton dimensions

            # If the mask is still 3D, it likely has multiple channels
            if mask.dim() == 3:
                # Option 1: Select the first channel (if appropriate)
                mask = mask[0, :, :]
                # Option 2: Average across channels to get a single 2D mask
                # mask = mask.mean(dim=0)

            if mask.dim() > 2:
                raise ValueError(f"Mask should be 2D, but got {mask.dim()}D after squeezing.")

            # Convert the mask to a 2D numpy array
            mask = mask.cpu().numpy()

        # Now mask should be a 2D numpy array
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=erosion)
        eroded_mask_binary = (eroded_mask / 255).astype(np.float32)

        # Convert back to a PyTorch tensor and move to the original device
        eroded_mask_tensor = torch.from_numpy(eroded_mask_binary).to(original_device)

        return eroded_mask_tensor

