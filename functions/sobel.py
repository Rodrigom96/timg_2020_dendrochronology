import cv2


def sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    phase = cv2.phase(sobelx, sobely, angleInDegrees=False)

    return grad, phase


def sobel_y(image, ksize=3):
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return abs_grad_y


def positive_sobel_x(img, ksize=3):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_x[grad_x < 0] = 0

    return grad_x
