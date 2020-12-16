import argparse

from functions.load_image import load_image
from functions.image_color import change_color_space
from functions.detect_center import detect_center
from functions.profile_measure import angles_mean_growth
from functions.error import relative_error

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str)

args = parser.parse_args()

def main(path):
    # load image
    image, metric = load_image(path)    

    # find center of tree
    gray = change_color_space(image)
    ray_radius = int(gray.shape[1]/5)
    center = detect_center(gray, ray_radius, debug=False)

    # meassure mean growth
    mean_growth_dict = angles_mean_growth(image, metric, center)
    
    print('\n Medias experimentales')
    print('\t N:',mean_growth_dict[270.0])
    print('\t S:',mean_growth_dict[90.0])
    print('\t E:',mean_growth_dict[0.0])
    print('\t W:',mean_growth_dict[180.0])

if __name__ == '__main__':
    args = parser.parse_args()
    image_path = args.image

    main(image_path)
