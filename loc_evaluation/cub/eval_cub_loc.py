#-*-coding:utf-8-*-

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_cub_gt_boxes(pre_path):

    current_file_path = os.path.dirname(os.path.abspath(__file__))

    id_path = os.path.join(current_file_path, "images.txt")

    gt_path = os.path.join(current_file_path, "bounding_boxes.txt")

    gt_test = []

    gt_boxes = []
    image_size = []

    f_gt = open(gt_path,"r")
    gt_all = f_gt.readlines()

    # get_boxes
    for gt_i in gt_all:

        gt_i = gt_i.split(" ")
        tmp = []
        for j in range(len(gt_i)):
            if j == 0:
                tmp.append(int(gt_i[0]))
            elif j == 1:
                tmp.append(float(gt_i[j]))
            elif j == 2:
                tmp.append(float(gt_i[j]))
            elif j == 3:
                tmp.append(float(gt_i[j]) + float(gt_i[1]))

            elif j == 4:
                tmp.append(float(gt_i[j]) + float(gt_i[2]))

        gt_test.append(tmp)

    f_gt.close()

    # get ids
    ids_all = []
    f_id = open(id_path, "r")
    ids = f_id.readlines()

    f_id.close()

    for id in ids:
        id = id.split(" ")
        ids_all.append(id)

    image_origin = []

    for pre_i in range(len(pre_path)):

        pre = pre_path[pre_i]
        image_tmp = cv2.imread(pre)

        image_origin.append(image_tmp)
        pre = pre.split("/")
        pre_last = "/".join(pre[4:])
        my_id = 0
        for id_j in range(len(ids_all)):
            tttm = ids_all[id_j][1].rstrip("\n")
            if tttm == pre_last:
                my_id = int(ids_all[id_j][0])
                break

        gt_boxes.append(gt_test[my_id-1][1:])

    return gt_boxes, image_origin
