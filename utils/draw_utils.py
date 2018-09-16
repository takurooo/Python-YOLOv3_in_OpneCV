#------------------------------------------------------
# import
#------------------------------------------------------
import os
import cv2

#------------------------------------------------------
# global
#------------------------------------------------------
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 10

FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
FONT_COLOR = (0, 0, 0) #BGR
FONT_BACKGROUND_COLOR = (0, 0, 255)

#------------------------------------------------------
# function
#------------------------------------------------------
def draw_results(img, classnames, scores, boxes):
    for classname, score, box in zip(classnames, scores, boxes):
        draw_result(img, classname, score, box)


def draw_result(img, classname, score, box):

    left, top, right, bottom = box
    cv2.rectangle(img, (left, top), (right, bottom),
                 LINE_COLOR, LINE_THICKNESS)

    label = '{}:{:.2f}'.format(classname, score)
    label_size, _ = cv2.getTextSize(label,
                                   FONT_STYLE, FONT_SCALE, FONT_THICKNESS)

    text_width, text_height = label_size

    cv2.rectangle(img, (left, top), (left+text_width, top+text_height),
                 FONT_BACKGROUND_COLOR, thickness=-1)
    cv2.putText(img, label, (left, top+text_height),
                FONT_STYLE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False)
