#-*-coding:utf-8-*
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from loc_evaluation.awa2.eval_awa2_loc import judge_awa2_image
from loc_evaluation.cub.eval_cub_loc import read_cub_gt_boxes


def get_cam(feat, class_attribute, label):

    cls_attr = class_attribute[label, :]
    c, h, w = feat.shape
    cam = np.dot(cls_attr.reshape((1, -1)), feat.reshape((c, -1))).reshape((h, w))
    return cam


def generate_pred_box(cam, iimage, _gt_bbox, db_name):

    ih = iimage.shape[0]

    iw = iimage.shape[1]

    threshold = 0.2
    # resize w*h
    cam = cv2.resize(cam, (iw, ih),
                     interpolation=cv2.INTER_CUBIC)
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    blend = iimage * 0.5 + heatmap_BGR * 0.5
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    thr_val = threshold * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap,
                                        int(thr_val), 255,
                                        cv2.THRESH_BINARY)

    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    blend_bbox = blend.copy()

    rect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect.append([x, y, w, h])
    if len(rect) == 0:
        predict_boxes = [0, 0, 1, 1]
    else:
        x, y, w, h = large_rect(rect)
        predict_boxes = [x, y, x + w, y + h]
        cv2.rectangle(blend_bbox,
                      (int(x), int(y)),
                      (int(x + w), int(y + h)),
                      (0, 0, 255), 2)

    if db_name == 'CUB':
        cv2.rectangle(blend_bbox,
                      (int(_gt_bbox[0]), int(_gt_bbox[1])),
                      (int(_gt_bbox[2]), int(_gt_bbox[3])),
                      (0, 255, 0), 2)
        return predict_boxes, blend_bbox, _gt_bbox

    elif db_name == 'AwA2':
        iou_all = []
        iou_max = 0
        inx_max = 0
        for gi in range(len(_gt_bbox)):
            iou = calculate_iou(predict_boxes, _gt_bbox[gi])
            iou_all.append(iou)
            if iou > iou_max:
                iou_max = iou
                inx_max = gi

        gt_box_only = _gt_bbox[inx_max]

        cv2.rectangle(blend_bbox,
                  (int(gt_box_only[0]), int(gt_box_only[1])),
                  (int(gt_box_only[2]), int(gt_box_only[3])),
                  (0, 255, 0), 2)

        return predict_boxes, blend_bbox, gt_box_only


def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    #cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]

    return intensity.astype('float32') * 255.0


def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    for i in range(len(rect)):
        area = rect[i][2] * rect[i][3]
        if large_area < area:
            large_area = area
            target = i

    x = rect[target][0]
    y = rect[target][1]
    w = rect[target][2]
    h = rect[target][3]

    return x, y, w, h


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def eval_c_gt_known(featmaps, pre_path, label_gt, db_name, class_attribute, setting):

    cnt = 0
    hit_correct = 0

    if db_name == 'AwA2':
        if setting == "unseen":
            option = 'unseen'
            gt_boxes, image_origin, inx_save = judge_awa2_image(pre_path, option)

            featmaps = featmaps[inx_save]
            label_gt = label_gt[inx_save]
        else:
            option = 'seen'
            gt_boxes, image_origin, inx_save = judge_awa2_image(pre_path, option)

            featmaps = featmaps[inx_save]
            label_gt = label_gt[inx_save]

    else:

        gt_boxes, image_origin = read_cub_gt_boxes(pre_path)

    for i in range(len(gt_boxes)):

        cnt = cnt + 1

        cam = get_cam(featmaps[i], class_attribute, label_gt[i])

        pred_box, blend, gt_box_ins = generate_pred_box(cam, image_origin[i], gt_boxes[i], db_name)

        iou = calculate_iou(gt_box_ins, pred_box)

        if iou >= 0.5:

            hit_correct = hit_correct + 1

    hit_correct = hit_correct * 1.0 / cnt

    print("hit_correct_gt_konwn: ", round(100 * hit_correct, 2))

    return hit_correct


def eval_c_gt_unknown(featmaps, pre_path, pred_labels, label_gt, db_name, class_attribute, setting):

    cnt = 0.0

    hit_top1 = 0.0
    hit_top2 = 0.0
    hit_top3 = 0.0
    hit_top4 = 0.0
    hit_top5 = 0.0

    if db_name == 'AwA2':
        if setting == "unseen":
            option = 'unseen'
            gt_boxes, image_origin, inx_save = judge_awa2_image(pre_path, option)

            featmaps = featmaps[inx_save]
            label_gt = label_gt[inx_save]
            pred_labels = pred_labels[inx_save]

        else:
            option = 'seen'
            gt_boxes, image_origin, inx_save = judge_awa2_image(pre_path, option)
            label_gt = label_gt[inx_save]
            featmaps = featmaps[inx_save]
            pred_labels = pred_labels[inx_save]

    else:

        gt_boxes, image_origin = read_cub_gt_boxes(pre_path)

    for i in range(pred_labels.shape[0]):
        cnt = cnt + 1

        for j in range(pred_labels.shape[1]):
            if pred_labels[i][j] == label_gt[i]:
                cam = get_cam(featmaps[i], class_attribute, pred_labels[i][j])
                pred_box, blend, gt_box_ins = generate_pred_box(cam, image_origin[i], gt_boxes[i],db_name)
                iou = calculate_iou(gt_box_ins, pred_box)
                if iou >= 0.5:
                    if j<=4:
                        hit_top5 = hit_top5 + 1

                    if j <=3:
                        hit_top4 = hit_top4 + 1

                    if j <= 2:
                        hit_top3 = hit_top3 + 1

                    if j <= 1:
                        hit_top2 = hit_top2 + 1

                    if j == 0:
                        hit_top1 = hit_top1 + 1

    hit_top1 = hit_top1 / cnt
    hit_top2 = hit_top2 / cnt
    hit_top3 = hit_top3 / cnt
    hit_top4 = hit_top4 / cnt
    hit_top5 = hit_top5 / cnt
    hit_ave = (hit_top1 + hit_top2 + hit_top3 + hit_top4 + hit_top5) / 5.0

    print("hit_top1: ", round(100 * hit_top1, 2))
    print("hit_top2: ", round(100 * hit_top2, 2))
    print("hit_top3: ", round(100 * hit_top3, 2))
    print("hit_top4: ", round(100 * hit_top4, 2))
    print("hit_top5: ", round(100 * hit_top5, 2))
    print("hit_ave: ", round(100 * hit_ave, 2))

    return [hit_top1, hit_top2, hit_top3, hit_top4, hit_top5, hit_ave]


def eval_g_gt_known(featmaps, pre_path, label_gt, db_name, class_attribute):

    if db_name == 'AwA2':
        seen_num = 5882
    else:
        seen_num = 1764

    featmaps_seen = featmaps[0:seen_num]
    pre_path_seen = pre_path[0:seen_num]
    label_gt_seen = label_gt[0:seen_num]

    featmaps_unseen = featmaps[seen_num:]
    pre_path_unseen = pre_path[seen_num:]
    label_gt_unseen = label_gt[seen_num:]
    u_setting = "unseen"
    setting = "seen"
    hit_correct_seen = eval_c_gt_known(featmaps_seen, pre_path_seen, label_gt_seen, db_name, class_attribute,setting)
    hit_correct_unseen = eval_c_gt_known(featmaps_unseen, pre_path_unseen, label_gt_unseen, db_name, class_attribute,u_setting)

    hit_h = 2 * hit_correct_seen * hit_correct_unseen / (hit_correct_seen + hit_correct_unseen)
    print("hit_gt_known_seen: ", round(hit_correct_seen * 100, 2), " hit_gt_known_unseen: ",
          round(100 * hit_correct_unseen, 2),
          " hit_gt_known_h: ", round(100 * hit_h, 2))


def eval_g_gt_unknown(featmaps, pre_path, pred_labels_seen, pred_labels_unseen, label_gt, db_name, class_attribute):
    if db_name == 'AwA2':
        seen_num = 5882
    else:
        seen_num = 1764

    featmaps_seen = featmaps[0:seen_num]
    pre_path_seen = pre_path[0:seen_num]
    label_gt_seen = label_gt[0:seen_num]

    featmaps_unseen = featmaps[seen_num:]
    pre_path_unseen = pre_path[seen_num:]
    label_gt_unseen = label_gt[seen_num:]
    u_setting = "unseen"
    setting = "seen"
    hit_correct_seen = eval_c_gt_unknown(featmaps_seen, pre_path_seen, pred_labels_seen,label_gt_seen, db_name, class_attribute, setting)
    hit_correct_unseen = eval_c_gt_unknown(featmaps_unseen, pre_path_unseen, pred_labels_unseen, label_gt_unseen, db_name, class_attribute,u_setting)
    hit_correct_seen = np.array(hit_correct_seen)
    hit_correct_unseen = np.array(hit_correct_unseen)

    hit_h = 2*hit_correct_seen*hit_correct_unseen/(hit_correct_seen + hit_correct_unseen)

    hit_h[5] = (hit_h[0] + hit_h[1] + hit_h[2] + hit_h[3] + hit_h[4]) / 5.0

    print("hit_top1_seen: ", round(100 * hit_correct_seen[0], 2), " hit_top1_unseen: ",
          round(100 * hit_correct_unseen[0], 2), " hit_top1_h: ", round(100 * hit_h[0], 2))

    print("hit_top2_seen: ", round(100 * hit_correct_seen[1], 2), " hit_top2_unseen: ",
          round(100 * hit_correct_unseen[1], 2), " hit_top2_h: ", round(100 * hit_h[1], 2))

    print("hit_top3_seen: ", round(100 * hit_correct_seen[2], 2), " hit_top3_unseen: ",
          round(100 * hit_correct_unseen[2], 2), " hit_top3_h: ", round(100 * hit_h[2], 2))

    print("hit_top4_seen: ", round(100 * hit_correct_seen[3], 2), " hit_top4_unseen: ",
          round(100 * hit_correct_unseen[3], 2), " hit_top4_h: ", round(100 * hit_h[3], 2))

    print("hit_top5_seen: ", round(100 * hit_correct_seen[4], 2), " hit_top5_unseen: ",
          round(100 * hit_correct_unseen[4], 2), " hit_top5_h: ", round(100 * hit_h[4], 2))

    print("hit_ave_seen: ", round(100 * hit_correct_seen[5], 2), " hit_ave_unseen: ", round(100 * hit_correct_unseen[1], 2),
          " hit_ave_h: ", round(100 * hit_h[5], 2))


def read_class_attribute(matrix_path):

    f_matrix = open(matrix_path, "r")
    matrix_all = f_matrix.readlines()
    class_attribute = []
    for i in range(len(matrix_all)):
        a = matrix_all[i].split(" ")
        tmp = []
        for j in range(len(a)):
            if a[j]:
                tmp.append(float(a[j]))

        class_attribute.append(tmp)

    class_attribute = np.array(class_attribute)

    f_matrix.close()

    return class_attribute
