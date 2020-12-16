import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.measure import profile_line
from skimage.filters import gaussian
import matplotlib.pyplot as plt

from functions.filters import median_filter
from functions.rectification import lineal_to_polar
from functions.sobel import positive_sobel_x, sobel_y
from util import log_time


def point_score(point, gray_image, roi_radius):
    '''
    Get score of point

    Args:
        point (tuple[int]): positon of point in image to evalueate
        gray_image (np.ndarray): gray image
        roi_radius (int): radius of search

    Returns:
        score (falot): score of point
    '''
    # convert gray image to polar
    angles = int(roi_radius/3)
    polar_image = lineal_to_polar(
        gray_image, point, radius=roi_radius, angles=angles)

    # calculate sobel of image in axis x and axis y
    polar_sobel_x = positive_sobel_x(polar_image)/255
    polar_sobel_y = sobel_y(polar_image)/255

    # calculte socre using sobel images
    score = np.sum(polar_sobel_x**2) - np.sum(polar_sobel_y**2)
    
    return score


def center_score_grid(gray_image, grid_center, lenght, roi_radius, size=15):
    '''
    Get center score of point in grid

    Args:
        gray_image (np.darray): Gray image
        grid_center (tuple(int)): Center position of grid in image
        length (int): length of grid
        roi_radius (int): point score search radius
        size (int): size of grid

    Returns:
        X (np.array): list of position in x axis of point in grid
        Y (np.array): list of position in y axis of point in grid
        Z (np.array): scores of grid
    '''
    # calculate point of grid position
    X = grid_center[0] + np.linspace(-lenght/2, lenght/2, size)
    Y = grid_center[1] + np.linspace(-lenght/2, lenght/2, size)

    # calculate socore of points in grid
    Z = np.zeros((size, size))
    future_scores = {}
    with ThreadPoolExecutor() as executor:
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                future_scores[executor.submit(
                    point_score, (x, y), gray_image, roi_radius)] = (i, j)
        for future in as_completed(future_scores):
            i, j = future_scores[future]
            score = future.result()
            Z[i, j] = score

    return X, Y, Z


#@log_time
def detect_center(gray_image, roi_radius, scales=2, debug=False):
    '''
    Find center of image

    Args:
        gray_image (np.array): Gray image
        roi_radius (int): radius of search
        scales (int) (optional): scales of search

    Returns:
        center (tuple): position of center
    '''
    # get image center
    img_h, img_w = gray_image.shape[:2]

    # iterate over soceres grid to find center
    grid_center = int(img_w/2), int(img_h/2)
    grid_length = int(np.sqrt(img_w**2 + img_h**2)/4)
    for s in range(scales):
        x, y, Z = center_score_grid(
            gray_image, grid_center, grid_length, roi_radius)

        if s != 0:
            Z = gaussian(Z, sigma=0.9)

        i, j = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
        grid_center = (x[i], y[j])
        grid_length = int(grid_length/16)

    if debug:
        # scores grid visualisation
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    center = int(grid_center[0]), int(grid_center[1])
    return center
