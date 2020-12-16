import cv2


def change_color_space(img):
    '''
    Change color sapce

    Args:
        img (np.darray): color image

    Returns:
        result (np.darrat): one channel image
    '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
