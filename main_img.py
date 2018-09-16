#------------------------------------------------------
# import
#------------------------------------------------------
import os
import argparse
import codecs
import time
import imghdr
import numpy as np
import cv2
print("opencv : ",cv2.__version__)
from model_wrapper import *
#------------------------------------------------------
# global
#------------------------------------------------------



#------------------------------------------------------
# function
#------------------------------------------------------
def process_images(model, img_path):

    img = cv2.imread(img_path)
    img_size = (img.shape[1], img.shape[0], img.shape[2])

    inference(model, img, draw=True)

    # display image
    cv2.imshow("image", img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # write image with bbox
    base_fname, _ = os.path.splitext(os.path.basename(img_path))
    out_file_path = os.path.join(os.getcwd(), base_fname+'_with_box.jpg')
    cv2.imwrite(out_file_path, img.astype(np.uint8));
    print("save : ", out_file_path)


def arg_parser():
    parser = argparse.ArgumentParser(description="Annotate with YOLOv3.")
    parser.add_argument("img_path", type=str, help="path2your_image", default=None)
    return parser


def main(args):
    process_images(load_model(), args.img_path)


#------------------------------------------------------
# main
#------------------------------------------------------
if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    main(args)
