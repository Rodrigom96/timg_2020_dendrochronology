import cv2
import numpy as np

from functions.rectification import lineal_to_polar
from functions.polar_scale import image_scales, image_scale_transform, get_valid_scales_regions
from functions.filters import median_filter
from functions.image_color import change_color_space
from functions.sobel import sobel, positive_sobel_x


def profile_measure(image, pixel_to_mm, angle_step=1):
    '''
    Meassure growth of tree ring in thresholdized polar image

    Args:  
        image (np.ndarray): thresholdized polar image
        pixel_to_mm (float): scale of image, to scale pixel to mm
        angle_step (int): angle meassure step

    Returns:
        mean_radial_growth (list): mean growth for angles
    '''
    mean_radial_growth = []  # lista con los promedios en cada direccion
    M, N = image.shape[:2]  # tamaño de la imagen
    # recorro las filas de la imagen polarizada, serian los angulos.
    for angle in range(0, M, angle_step):
        growth = 0
        growths_ring = []
        for i in range(len(image[angle, :])):  # recorro la fila
            if image[angle, i] > 0:  # cuando sale de 0 reinicio el contador de medida
                if growth != 0:  # si sigo arriba de cero no agrego el valor.
                    growths_ring.append(growth)
                growth = 0
            else:
                growth += 1
        mean_growth = np.mean(growths_ring) * \
            pixel_to_mm / 2  # promedio y paso a mm
        mean_radial_growth.append(mean_growth)  # anexo a la salida.
    return mean_radial_growth


def filter_mean_growth_angles_dict(mean_growth_dict, alpha=0.2):
    '''
    Apply low pass filter mean growth angles measure

    Args:
        mean_growth_dict (dict): dictionary of mean grwoth for angles
        alpha (float): low pass filter factor

    Returns:
        filter_mean_growth_dict(dict): dictionary of mean grwoth filtered
    '''
    assert 0 < alpha <= 1

    filter_mean_growth_dict = {}
    growth_angles = np.sort(list(mean_growth_dict.keys()))
    for i in range(len(growth_angles)):
        if i == 0:
            a0 = growth_angles[-1]
            a1 = growth_angles[i]
            a2 = growth_angles[i+1]
        elif i != len(growth_angles) - 1:
            a1 = growth_angles[i-1]
            a1 = growth_angles[i]
            a2 = growth_angles[i+1]
        else:
            a1 = growth_angles[i-1]
            a1 = growth_angles[i]
            a2 = growth_angles[0]

        beta = (1 - alpha)/2
        filter_mean_growth_dict[a1] = beta*mean_growth_dict[a0] + \
            alpha*mean_growth_dict[a1] + beta*mean_growth_dict[a2]

    return filter_mean_growth_dict


def angles_mean_growth(image, pixel_to_mm, center):
    '''
    Get mean growth for angles

    Args:
        image (np.darray): numpy color image
        ppixel_to_mm (float): scale to convert pixel distance in mm
        center (tuple[int]): center of tree in image

        Returns:
        mean_growth_dict (dict): dictionary of mean grwoth for angles
    '''
    assert len(image.shape) == 3 and image.shape[2] == 3

    ANGLES = 720
    angle_step = 5

    # get polar image
    polarized_image = lineal_to_polar(image, center, angles=ANGLES)
    scale_find_radius = int(polarized_image.shape[1]/3)
    scales = image_scales(polarized_image, angle_step, scale_find_radius)

    # find mean growth of valid regions
    valid_scales = get_valid_scales_regions(scales)
    mean_growth_dict = {}
    for x1, x2 in valid_scales:
        # crop polar image in region
        polar_region_ori = polarized_image[angle_step*x1:angle_step*x2]

        # apply scale tranform
        region_scales = scales[x1:x2]
        region_scales = np.array(region_scales)/np.min(region_scales)
        polar_region = image_scale_transform(
            polar_region_ori, region_scales, angle_step)

        # appy filter in y
        polar_region = median_filter(polar_region, (15, 3)).astype(np.uint8)
        polar_region = cv2.fastNlMeansDenoisingColored(
            polar_region, 10, 11, 21)
        polar_region = change_color_space(polar_region)

        # apply sobel in x axis, to get end of rings
        sobel_image = positive_sobel_x(polar_region, ksize=3)
        sobel_image = median_filter(sobel_image, kernel=7)
        th2 = sobel_image > np.mean(sobel_image)*2

        # meassure mean growth
        mean_growth = profile_measure(th2, pixel_to_mm, angle_step=angle_step)

        # rescale mean grwoth
        mean_growth = np.array(mean_growth) / region_scales

        # save mean growth
        for i, m in enumerate(mean_growth):
            mean_growth_dict[360*angle_step*(x1 + i)/ANGLES] = m

    # find mean grwoth of not valid regions
    for a in np.arange(0, 360 - 360*angle_step/ANGLES, 360*angle_step/ANGLES):
        if mean_growth_dict.get(a):
            continue

        y1 = int(ANGLES*a/360)
        y2 = int(ANGLES*a/360 + angle_step)
        polar_region = polarized_image[y1:y2]
        polar_region = median_filter(polar_region, (3, 3)).astype(np.uint8)
        polar_region = cv2.fastNlMeansDenoisingColored(
            polar_region, 10, 11, 21)
        polar_region = change_color_space(polar_region)

        sobel_image = positive_sobel_x(polar_region, ksize=3)
        sobel_image = median_filter(sobel_image, kernel=7)
        th2 = sobel_image > np.mean(sobel_image)*2

        mean_growth = profile_measure(
            th2, pixel_to_mm, angle_step=angle_step)[0]
        mean_growth_dict[a] = mean_growth

    # aplly low pass filter to mean growths
    mean_growth = filter_mean_growth_angles_dict(mean_growth_dict)

    return mean_growth_dict
