#------------------------------------------------------
# import
#------------------------------------------------------
import os
import argparse
import cv2
import numpy as np


#------------------------------------------------------
# global
#------------------------------------------------------


#------------------------------------------------------
# function
#------------------------------------------------------
class YOLO_OpenCV():

    def __init__(self, config='yolov3.cfg', weights='yolov3.weights',
                 classfile='coco.names', width=416, height=416, score_th=0.7, nms_th=0.4):

        self.in_size = (width, height)
        self.height = height
        self.score_th = score_th
        self.nms_th = nms_th
        self.model = cv2.dnn.readNetFromDarknet(config, weights)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(classfile, 'r') as f:
            self.classes = f.read().rstrip('\n').split('\n')


    def fit(self, img):
        # (width, height, channel) -> (1, channel, width, height)
        # pixel = pixel * scalefactor
        # pixel = pixel - mean[R,G,B]
        # if swapRB is True, BGR -> RGB
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=self.in_size,
                                     mean=[0,0,0], swapRB=True, crop=False)

        self.model.setInput(blob)
        outs = self.model.forward(self._outlayers(self.model))
        return self._postprocess(img, outs)

    def _to_classname(self, classid):
        if self.classes:
            assert(classid < len(self.classes))
        return self.classes[classid]

    def _outlayers(self, model):
        layersNames = self.model.getLayerNames()
        output_layers = [layersNames[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        return output_layers

    def _postprocess(self, img, outs):
        img_height, img_width, _ = img.shape

        classids = []
        scores = []
        boxes = []
        for out in outs:
            for detection in out:
                # detection =
                # [center_x, center_y, box_width, box_height,
                #  confidence(box enclose object), confidence(each class)....]
                detected_scores = detection[5:]
                classid = np.argmax(detected_scores)
                score = detected_scores[classid]
                if score > self.score_th:
                    center_x = int(detection[0] * img_width)
                    center_y = int(detection[1] * img_height)
                    box_width = int(detection[2] * img_width)
                    box_height = int(detection[3] * img_height)
                    left = max(0, int(center_x - box_width/2))
                    top = max(0, int(center_y - box_height/2))
                    classids.append(classid)
                    scores.append(float(score))
                    boxes.append([left, top, box_width, box_height])

        out_classes = []
        out_scores = []
        out_boxes = []
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.score_th, self.nms_th)

        for i in indices:
            i = i[0]
            left, top, box_width, box_height = boxes[i]

            left = max(0, np.floor(left+0.5).astype(np.uint32))
            top = max(0, np.floor(top+0.5).astype(np.uint32))
            right = min(img_width, np.floor(left+box_width+0.5).astype(np.uint32))
            bottom = min(img_height, np.floor(top+box_height+0.5).astype(np.uint32))

            out_classes.append(self._to_classname(classids[i]))
            out_scores.append(scores[i])
            out_boxes.append([left, top, right, bottom])
        return out_classes, out_scores, out_boxes
