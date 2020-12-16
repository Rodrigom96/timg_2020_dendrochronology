from scipy import ndimage


def median_filter(image, kernel):
    '''
    Apply median filter to image

    Args:
        image (np.darray): image
        kernel (int, tuple): size of kernel

    Returns:
        result (np.darray): filtered image
    '''
    ch = image.shape[2] if len(image.shape) > 2 else 1
    
    if isinstance(kernel, int):
        if ch > 2:
            size = (kernel, kernel, 1)
        else:
            size = (kernel, kernel)
    elif isinstance(kernel, tuple):
        if len(kernel) < ch:
            size = kernel + (1,)*(ch - len(kernel))
        else:
            size = kernel
    else:
        size = kernel

    return ndimage.median_filter(image, size=size)
