import cv2
import numpy as np

def find_scale(row1, row2):
    '''
    Find best scale in x axis between row for mach

    Args:
        row1 (np.darray): image to mach
        row2 (np.darray): image that scale for match

    Returns:
        s (float): scale
    '''
    scores = []
    scales = []
    for scale in range(900, 1100):
        s = scale/1000
        tmp_r = cv2.resize(row2, (round(s*row2.shape[1]), row2.shape[0]))
        tmp2 = np.zeros_like(row1)
        tmp2[:, :min(row2.shape[1], tmp_r.shape[1])] = tmp_r[:,
                                                             :min(row2.shape[1], tmp_r.shape[1])]
        score = abs(np.mean(tmp2[:, :min(row2.shape[1], tmp_r.shape[1])]) -
                    np.mean(row1[:, :min(row2.shape[1], tmp_r.shape[1])]))**2

        t1 = row1[:, :min(row2.shape[1], tmp_r.shape[1])]
        t2 = tmp2[:, :min(row2.shape[1], tmp_r.shape[1])]
        t1 = t1/np.sqrt(np.sum(t1**2))
        t2 = t2/np.sqrt(np.sum(t2**2))

        score = np.sum(t1*t2)

        scores.append(score)
        scales.append(s)

    s = np.array(scales)[np.argmax(scores)]
    return s


def image_scales(polar_image, step, radius):
    '''
    Find scales of row in image to aling rings

    Args:
        polar_image (np.darray): polar image
        step (int): rows heights
        radius (int): radius of search

    Returns:
        scales (list[float]): list of scales
    '''
    scales = [1]
    for i in range(0, int((polar_image.shape[0]//step)) - 1):
        row1 = polar_image[step*i:step*(i+1), :radius]
        row2 = polar_image[step*(i+1):step*(i+2), :radius]

        scale = find_scale(row1, row2)
        scales.append(scale)

    return scales


def image_scale_transform(image, scales, step):
    '''
    Scale image by rows

    Args:
        image (np.darray): image
        scales (list[float]): scales of rows
        step (int): row heights

    Returns:
        result (np.darray): scaled image
    '''
    result = np.zeros((0, ) + image.shape[1:3], dtype=image.dtype)
    for i, s in enumerate(scales):
        # get row
        row = image[step*i:step*(i+1)]

        # scale row
        row_scaled = cv2.resize(row, (round(s*row.shape[1]), row.shape[0]))
        tmp = np.zeros_like(row)
        tmp[:, :min(tmp.shape[1], row_scaled.shape[1])
            ] = row_scaled[:, :min(tmp.shape[1], row_scaled.shape[1])]
        result = np.vstack((result, tmp))

    return result


def get_valid_scales_regions(scales, min_lenght=5):
    '''
    Find valid regions of scales, if scales not change a lot between rows

    Args:
        scales (list[float]): scales
        min_lenght (int): min continus rows with no gaps in scales

    Returns:
        valid_regions_indexs (list[tuples]): list of tuples of index in regions scales
    '''
    # find scales gap
    scales_gap_index = [0]
    for i in range(1, len(scales)-1):
        last_s = scales[scales_gap_index[-1]]
        s = scales[i]
        if abs(s - last_s) > 0.05:
            scales_gap_index.append(i)

    # find valid regions
    valid_regions_indexs = []
    for i in range(0, len(scales_gap_index) - 1):
        idx1 = scales_gap_index[i]
        idx2 = scales_gap_index[i+1]

        if idx2 - idx1 > min_lenght:
            valid_regions_indexs.append((idx1, idx2))

    return valid_regions_indexs
