import cv2
import exifread


def load_image(route):
    '''
    Load image from route

    Args:
        route (str): route of image

    Returns:
        img (np.darray): numpy image
        metric (float): pixel to mm scale
    '''
    img = cv2.imread(route)

    with open(route, 'rb') as f:
        tags = exifread.process_file(f)

    # la constante es para pasar a milimetros los DPI
    metric = 25.4/tags['Image XResolution'].values[0]
    return img, metric
