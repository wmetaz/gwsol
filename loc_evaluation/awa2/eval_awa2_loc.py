#-*-coding:utf-8-*-

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


##get object annotation bndbox loc start
def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC
    tree = ET.ElementTree(file=AnotPath)
    root = tree.getroot()
    box_all = []

    for obj in root.iter('object'):
        for item in obj.iter('item'):
            xml_box = item.find('bndbox')
            if float(xml_box.find('xmin').text) <0:

                xmin = 0.0
            else:

                xmin = (float(xml_box.find('xmin').text))

            if float(xml_box.find('ymin').text) < 0:
                ymin = 0.0
            else:
                ymin = (float(xml_box.find('ymin').text))

            if float(xml_box.find('xmax').text) < 0:
                xmax = 0.0
            else:
                xmax = (float(xml_box.find('xmax').text))

            if float(xml_box.find('ymax').text) < 0:
                ymax = 0.0
            else:
                ymax = (float(xml_box.find('ymax').text))

            BndBoxLoc = [xmin, ymin, xmax, ymax]

            box_all.append(BndBoxLoc)

    return box_all


def judge_awa2_image(pre_path, option):

    current_file_path = os.path.dirname(os.path.abspath(__file__))

    gt_boxes = []

    image_origin = []
    inx_save = []

    if option == "unseen":
        for pre_i in range(len(pre_path)):
            pre = pre_path[pre_i]

            # get gt box
            xml_dir = os.path.join(current_file_path, "test_unseen_gt")
            pre_fix = pre.split("/")
            im_class = pre_fix[4]
            xml_path = xml_dir + "/" + im_class + "/" + pre_fix[5][:-4] + ".xml"

            if os.path.exists(xml_path):
                res = GetAnnotBoxLoc(xml_path)
                if res:
                    gt_boxes.append(res)
                    image_origin.append(cv2.imread(pre))
                    inx_save.append(pre_i)

    elif option == 'seen':

        for pre_i in range(len(pre_path)):
            pre = pre_path[pre_i]
            # get gt box
            xml_dir = os.path.join(current_file_path, "test_seen_gt", 'gt')
            pre_fix = pre.split("/")

            xml_path = xml_dir + "/" + pre_fix[5][:-4] + ".xml"
            if os.path.exists(xml_path):
                res = GetAnnotBoxLoc(xml_path)
                rec = res[0]
                w = rec[2] - rec[0]
                h = rec[3] - rec[1]

                if res and (w*h > 0):
                    gt_boxes.append(res)
                    image_origin.append(cv2.imread(pre))
                    inx_save.append(pre_i)

    return gt_boxes, image_origin, inx_save


