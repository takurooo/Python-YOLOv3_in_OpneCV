#------------------------------------------------------
# import
#------------------------------------------------------
import os
import argparse
import time
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
def process_webcam(model):
    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) < 0:
        ret, img = cap.read()

        if not ret:
            print("done.")
            break

        inference(model, img, draw=True)
        cv2.imshow("video", img)

    cap.release()
    cv2.destroyAllWindows()


def arg_parser():
    parser = argparse.ArgumentParser(description="view webcam.")
    return parser


def main(args):
    process_webcam(load_model())


#------------------------------------------------------
# main
#------------------------------------------------------
if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    main(args)
