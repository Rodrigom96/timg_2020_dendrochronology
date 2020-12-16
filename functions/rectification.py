import cv2
import numpy as np


def lineal_to_polar(image, center, radius=None, angles=None):
    '''
    Convert lineal image in cartecian coridante to polar image

    Args:
        image (np.ndarray): numpy image
        center (tuple[int]): center of polar tranform
        radius (int) (optional): max radius of polar transform, default image with
        angles (int) (optional): height/angles resoulution of polar image, default image height

    Returns:
        polar_image (np.ndarray): polar image
    '''
    if not radius:
        radius = np.sqrt(
            ((image.shape[0]/2.0)**2.0)+((image.shape[1]/2.0)**2.0))

    polar_image = cv2.linearPolar(image, center, radius, cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
    
    if angles:
        polar_image = cv2.resize(polar_image, (int(radius), angles), cv2.INTER_CUBIC)
    else:
        polar_image = cv2.resize(polar_image, (int(radius), polar_image.shape[0]))
    

    return polar_image
