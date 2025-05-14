import numpy as np
import torch
import torch.nn as nn

from partialconv2d import PartialConv2d
import scipy.ndimage


class encoder4(nn.Module):
    def __init__(self):
        super(encoder4, self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = PartialConv2d(3, 3, 1, 1, 0, return_mask=True)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = PartialConv2d(3, 64, 3, 1, 0, return_mask=True)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = PartialConv2d(64, 64, 3, 1, 0, return_mask=True)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = PartialConv2d(64, 128, 3, 1, 0, return_mask=True)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = PartialConv2d(128, 128, 3, 1, 0, return_mask=True)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = PartialConv2d(128, 256, 3, 1, 0, return_mask=True)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = PartialConv2d(256, 256, 3, 1, 0, return_mask=True)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = PartialConv2d(256, 256, 3, 1, 0, return_mask=True)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = PartialConv2d(256, 256, 3, 1, 0, return_mask=True)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = PartialConv2d(256, 512, 3, 1, 0, return_mask=True)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x, mask=None, sF=None, matrix11=None, matrix21=None, matrix31=None, cookie_cutter=False):
        output = {}

        out, mask = self.conv1(x, mask, cookie_cutter)
        out = self.reflecPad1(out);
        mask = self.reflecPad1(mask)
        out, mask = self.conv2(out, mask, cookie_cutter)
        output['r11'] = self.relu2(out)

        out = self.reflecPad7(output['r11']);
        mask = self.reflecPad7(mask)

        out, mask = self.conv3(out, mask, cookie_cutter)
        output['r12'] = self.relu3(out)

        output['p1'] = self.maxPool(output['r12']);
        mask = self.maxPool(mask)
        out = self.reflecPad4(output['p1']);
        mask = self.reflecPad4(mask)
        out, mask = self.conv4(out, mask, cookie_cutter)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21']);
        mask = self.reflecPad7(mask)

        out, mask = self.conv5(out, mask, cookie_cutter)
        output['r22'] = self.relu5(out)

        output['p2'] = self.maxPool2(output['r22']);
        mask = self.maxPool2(mask)
        out = self.reflecPad6(output['p2']);
        mask = self.reflecPad6(mask)
        out, mask = self.conv6(out, mask, cookie_cutter)
        output['r31'] = self.relu6(out)
        if (matrix31 is not None):
            feature3, transmatrix3 = matrix31(output['r31'], sF['r31'])
            out = self.reflecPad7(feature3);
            mask = self.reflecPad7(mask)
        else:
            out = self.reflecPad7(output['r31']);
            mask = self.reflecPad7(mask)
        out, mask = self.conv7(out, mask, cookie_cutter)
        output['r32'] = self.relu7(out)

        out = self.reflecPad8(output['r32']);
        mask = self.reflecPad8(mask)
        out, mask = self.conv8(out, mask, cookie_cutter)
        output['r33'] = self.relu8(out)

        out = self.reflecPad9(output['r33']);
        mask = self.reflecPad9(mask)
        out, mask = self.conv9(out, mask, cookie_cutter)
        output['r34'] = self.relu9(out)

        output['p3'] = self.maxPool3(output['r34']);
        mask = self.maxPool3(mask)
        out = self.reflecPad10(output['p3']);
        mask = self.reflecPad10(mask)
        out, mask = self.conv10(out, mask, cookie_cutter)
        output['r41'] = self.relu10(out)
        return output, mask


class decoder4(nn.Module):
    def __init__(self):
        super(decoder4, self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = PartialConv2d(512, 256, 3, 1, 0, return_mask=True)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)  # For the masks

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = PartialConv2d(256, 256, 3, 1, 0, return_mask=True)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = PartialConv2d(256, 256, 3, 1, 0, return_mask=True)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = PartialConv2d(256, 256, 3, 1, 0, return_mask=True)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = PartialConv2d(256, 128, 3, 1, 0, return_mask=True)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = PartialConv2d(128, 128, 3, 1, 0, return_mask=True)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = PartialConv2d(128, 64, 3, 1, 0, return_mask=True)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = PartialConv2d(64, 64, 3, 1, 0, return_mask=True)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = PartialConv2d(64, 3, 3, 1, 0, return_mask=True)

    def forward(self, x, mask=None, original_features=None, original_image=None, feathering=False, cookie_cutter=False):

        # if mask is not None:
        #    mask4 = mask
        #    mask3 = self.maxPool(mask4)
        #    mask2 = self.maxPool(mask3)
        #    mask1 = self.maxPool(mask2)
        # else:
        #    mask1 = None
        #    mask2 = None
        #    mask3 = None
        #    mask4 = None

        if mask is not None:
            # To avoid aliasing, the intermediate masks are based on the original mask
            _, _, rows, cols = x.shape
            # mask = mask.unsqueeze(0)
            mask1 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
            mask1 = (mask1 > 0.5).float()

            rows, cols = rows * 2, cols * 2
            mask2 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
            mask2 = (mask2 > 0.5).float()

            rows, cols = rows * 2, cols * 2
            mask3 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
            mask3 = (mask3 > 0.5).float()

            rows, cols = rows * 2, cols * 2
            mask4 = nn.functional.interpolate(mask, size=(rows, cols), mode='bilinear')
            mask4 = (mask4 > 0.5).float()

            mask1 = mask1.squeeze(0)
            mask2 = mask2.squeeze(0)
            mask3 = mask3.squeeze(0)
            mask4 = mask4.squeeze(0)
        else:
            mask1 = None
            mask2 = None
            mask3 = None
            mask4 = None

        # decoder

        # LAYER 1_1
        # Process masked features with mask
        out = self.reflecPad11(x)
        out11 = out
        if original_features is not None:
            out_org = self.reflecPad11(original_features['r41'])
            out12 = out_org
        else:
            out12 = out

        mask1 = self.reflecPad11(mask1) if mask1 is not None else None
        mask1 = penetrate_mask(mask1, 11) if feathering else mask1

        out = mask1 * out + (1 - mask1) * out_org if original_features is not None else out
        out13 = out

        mask_conv11 = mask1
        out, mask1 = self.conv11(out, mask1, cookie_cutter)
        out = self.relu11(out)
        out = self.unpool(out)

        # LAYER 2_1
        out = self.reflecPad12(out)
        if original_features is not None:
            out_org = self.reflecPad12(original_features['r34'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask2 = self.reflecPad12(mask2) if mask2 is not None else None
        mask2 = penetrate_mask(mask2, 11) if feathering else mask2

        out = mask2 * out + (1 - mask2) * out_org if original_features is not None else out

        mask_conv12 = mask2
        out, mask2 = self.conv12(out, mask2, cookie_cutter)
        out = self.relu12(out)

        # LAYER 2_2
        out = self.reflecPad13(out);
        if original_features is not None:
            out_org = self.reflecPad13(original_features['r33'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask2 = self.reflecPad13(mask2) if mask2 is not None else None
        mask2 = penetrate_mask(mask2, 11) if feathering else mask2

        out = mask2 * out + (1 - mask2) * out_org if original_features is not None else out

        mask_conv13 = mask2
        out, mask2 = self.conv13(out, mask2, cookie_cutter)
        out = self.relu13(out)

        # LAYER 2_3
        out = self.reflecPad14(out);
        if original_features is not None:
            out_org = self.reflecPad14(original_features['r32'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask2 = self.reflecPad14(mask2) if mask2 is not None else None
        mask2 = penetrate_mask(mask2, 11) if feathering else mask2

        out = mask2 * out + (1 - mask2) * out_org if original_features is not None else out

        mask_conv14 = mask2
        out, mask2 = self.conv14(out, mask2, cookie_cutter)
        out = self.relu14(out)
        out14 = out

        # LAYER 2_4
        out = self.reflecPad15(out);
        if original_features is not None:
            out_org = self.reflecPad15(original_features['r31'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask2 = self.reflecPad15(mask2) if mask2 is not None else None
        mask2 = penetrate_mask(mask2, 11) if feathering else mask2

        out = mask2 * out + (1 - mask2) * out_org if original_features is not None else out

        mask_conv15 = mask2
        out, mask2 = self.conv15(out, mask2, cookie_cutter)
        out = self.relu15(out)
        out15 = out
        out = self.unpool2(out)

        # LAYER 3_1
        out = self.reflecPad16(out);
        if original_features is not None:
            out_org = self.reflecPad16(original_features['r22'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask3 = self.reflecPad16(mask3) if mask3 is not None else None
        mask3 = penetrate_mask(mask3, 11) if feathering else mask3

        out = mask3 * out + (1 - mask3) * out_org if original_features is not None else out

        mask_conv16 = mask3
        out, mask3 = self.conv16(out, mask3, cookie_cutter)
        out = self.relu16(out)
        out16 = out

        # LAYER 3_2
        out = self.reflecPad17(out);
        if original_features is not None:
            out_org = self.reflecPad17(original_features['r21'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask3 = self.reflecPad17(mask3) if mask3 is not None else None
        mask3 = penetrate_mask(mask3, 11) if feathering else mask3

        out = mask3 * out + (1 - mask3) * out_org if original_features is not None else out

        mask_conv17 = mask3
        out, mask3 = self.conv17(out, mask3, cookie_cutter)
        out = self.relu17(out)
        out17 = out
        out = self.unpool3(out)

        # LAYER 4_1
        out = self.reflecPad18(out);
        if original_features is not None:
            out_org = self.reflecPad18(original_features['r12'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask4 = self.reflecPad18(mask4) if mask4 is not None else None
        mask4 = penetrate_mask(mask4, 11) if feathering else mask4

        out = mask4 * out + (1 - mask4) * out_org if original_features is not None else out

        mask_conv18 = mask4
        out, mask4 = self.conv18(out, mask4, cookie_cutter)
        out = self.relu18(out)
        out18 = out

        # LAYER 4_2
        out = self.reflecPad19(out);
        if original_features is not None:
            out_org = self.reflecPad19(original_features['r11'])
            min_h = min(out.size(2), out_org.size(2))
            min_w = min(out.size(3), out_org.size(3))
            out = out[:, :, :min_h, :min_w]
            out_org = out_org[:, :, :min_h, :min_w]

        mask4 = self.reflecPad19(mask4) if mask4 is not None else None
        mask4 = penetrate_mask(mask4, 11) if feathering else mask4

        out = mask4 * out + (1 - mask4) * out_org if original_features is not None else out

        mask_conv19 = mask4
        out, mask4 = self.conv19(out, mask4, cookie_cutter)
        out19 = out

        return out, mask_conv11, mask_conv12, mask_conv13, mask_conv14, mask_conv15, \
            mask_conv16, mask_conv17, mask_conv18, mask_conv19, \
            out11, out12, out13, out14, out15, out16, out17, out18, out19

    #def penetrate_mask(self, mask, radius):
    #    """
    #    Gradually penetrate into the unmasked area from the edges of the mask.

    #    Parameters:
    #    - mask (torch.Tensor): A 4D tensor (B, 1, H, W) containing 0s and 1s representing the mask.
    #    - radius (int): The number of pixels over which to transition from masked (1) to unmasked (0).

    #    Returns:
    #    - torch.Tensor: A new mask with gradual transition at the edges.
    #    """
    #    # Ensure the mask is a float tensor for gradual transition
    #    mask = mask.float()

    #    # Convert the mask to a NumPy array to use scipy's distance transform
    #    mask_np = mask.squeeze(0).cpu().numpy()  # Assuming the mask is [1, H, W]

    #    # Create a distance map from the edges of the mask
    #    # Distance transform: distance to the nearest zero pixel (unmasked area)
    #    distance_map = scipy.ndimage.distance_transform_edt(mask_np == 0)

    #    # Normalize the distance map to the specified radius
    #    penetration_mask_np = np.clip((radius - distance_map) / radius, 0, 1)

    #    # Convert the penetration mask back to a torch tensor
    #    penetration_mask = torch.tensor(penetration_mask_np, device=mask.device, dtype=mask.dtype).unsqueeze(0)

    #    return penetration_mask


def penetrate_mask(mask, radius, shrink=1):
    """
    Gradually penetrate into the unmasked area from the edges of the mask,
    after shrinking the mask by 'shrink' pixels.

    Parameters:
    - mask (torch.Tensor): A tensor of shape [1, H, W] containing 0s and 1s representing the mask.
    - radius (int): The number of pixels over which to transition from masked (1) to unmasked (0).
    - shrink (int): The number of pixels to shrink the mask before applying the penetration effect.

    Returns:
    - torch.Tensor: A new mask with gradual transition at the edges, shape [1, H, W].
    """
    import scipy.ndimage  # Ensure scipy.ndimage is imported
    import numpy as np

    # Ensure the mask is a float tensor for gradual transition
    mask = mask.float()

    # Convert the mask to a NumPy array to use scipy's functions
    mask_np = mask.squeeze(0).cpu().numpy()  # Shape: [H, W]

    # Shrink the mask by 'shrink' pixels using binary erosion
    if shrink > 0:
        eroded_mask_np = scipy.ndimage.binary_erosion(mask_np, iterations=shrink).astype(mask_np.dtype)
    else:
        eroded_mask_np = mask_np

    # Create a distance map from the edges of the eroded mask
    distance_map = scipy.ndimage.distance_transform_edt(eroded_mask_np == 0)

    # Normalize the distance map to the specified radius
    penetration_mask_np = np.clip((radius - distance_map) / radius, 0, 1)

    # Convert the penetration mask back to a torch tensor
    penetration_mask = torch.tensor(penetration_mask_np, device=mask.device, dtype=mask.dtype).unsqueeze(0)

    return penetration_mask  # Shape: [1, H, W]


