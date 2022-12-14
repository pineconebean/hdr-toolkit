import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


def read_ldr(path, bit, scale_factor=1.):
    if scale_factor == 1:
        return F.to_tensor(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                           .astype(np.float32) / (2 ** bit - 1))
    else:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        # reference: https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
        inter_method = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), interpolation=inter_method)
        return F.to_tensor(img.astype(np.float32) / (2 ** bit - 1))


def gamma_correction(img, exposure, gamma):
    return (img ** gamma) * (2.0 ** (-1 * exposure))


def ev_align(img, exposure, gamma):
    return ((img ** gamma) / (2.0 ** exposure)) ** (1 / gamma)


def read_tiff(path, exposure, cat=True, is_uint8=True):
    if is_uint8:
        # the image is already uint8
        img = F.to_tensor(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
    else:
        # cv2.imread without IMREAD_UNCHANGED will convert data to uint8 automatically
        img = F.to_tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    img_corrected = gamma_correction(img, exposure, 2.2)
    if cat:
        return torch.cat((img, img_corrected), dim=0)
    else:
        return img_corrected


def imread_uint16_png(image_path, alignratio_path):
    """ This function loads a uint16 png image from the specified path and restore its original image range with
    the ratio stored in the specified alignratio.npy respective path.


    Args:
        image_path (str): Path to the uint16 png image
        alignratio_path (str): Path to the alignratio.npy file corresponding to the image

    Returns:
        np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    # Load the align_ratio variable and ensure is in np.float32 precision
    align_ratio = np.load(alignratio_path).astype(np.float32)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio


def imwrite_uint16_png(image_path, image, alignratio_path):
    """ This function writes the hdr image as a uint16 png and stores its related align_ratio value in the specified paths.

        Args:
            image_path (str): Write path to the uint16 png image (needs to finish in .png, e.g. 0000.png)
            image (np.ndarray): HDR image in float format.
            alignratio_path (str): Write path to the align_ratio value (needs to finish in .npy, e.g. 0000_alignratio.npy)

        Returns:
            np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    align_ratio = (2 ** 16 - 1) / image.max()
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))
    return None


def imread_uint16_tiff(image_path):
    """ This function loads a uint16 tiff image from the specified path.
        Args:
            image_path (str): Path to the uint16 tiff image
        Returns:
            np.ndarray (np.uint16, (h,w,3)): Returns the RGB HDR image specified in image_path.
    """
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
