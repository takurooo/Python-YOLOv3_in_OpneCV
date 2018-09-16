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
SKIP_FRAME_NUM = 10



#------------------------------------------------------
# function
#------------------------------------------------------
def process_video(model, video_path):

    cap = cv2.VideoCapture(video_path)

    """
    base, _ = os.path.splitext(video_path)
    out_path = base + '_yolo_out_py.avi'
    video_writer = cv2.VideoWriter(out_path,
                                   cv2.VideoWriter_fourcc('M','J','P','G'),
                                   30,
                                   (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    """

    framecnt = 0
    while (cv2.waitKey(1)&0xFF) != ord('q'):
        framecnt += 1
        ret, img = cap.read()

        if not ret:
            print("done.")
            break

        if not (framecnt % SKIP_FRAME_NUM == 0):
            continue
        else:
            framecnt = 0

        inference(model, img, draw=True)
        cv2.imshow("video", img)

        #video_writer.write(img.astype(np.uint8))


    cap.release()
    cv2.destroyAllWindows()


def arg_parser():
    parser = argparse.ArgumentParser(description="View video.")
    parser.add_argument("video", type=str, help="path2your_video", default=None)
    return parser


def main(args):
    process_video(load_model(), args.video)


#------------------------------------------------------
# main
#------------------------------------------------------
if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    main(args)
