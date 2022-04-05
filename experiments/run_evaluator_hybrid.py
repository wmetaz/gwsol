import os
import os.path as osp
import sys
import argparse

import numpy as np 
import torch
from easydict import EasyDict as edict

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))
from util import *
from configs.default import cfg, update_datasets
from evaluation.utils import load_model, parse_checkpoint_paths, inference, crop_voting, report_evaluation, croptimes
from evaluation.hybrid import hybrid_labeling

from loc_evaluation.utils_loc import eval_c_gt_known, eval_c_gt_unknown, eval_g_gt_known, eval_g_gt_unknown


def preprocess_ss(cfg, all_cls_names):
    seen_cls = loadtxt(cfg.ss_train)
    seen_cls_indice = []

    unseen_cls = loadtxt(cfg.ss_test)
    unseen_cls_indice = []
    for cls in unseen_cls:
        unseen_cls_indice.append(all_cls_names.index(cls))
    for cls in seen_cls:
        seen_cls_indice.append(all_cls_names.index(cls))

    train_paths = []
    test_paths  = []

    for cls_indice, cls in enumerate(seen_cls):
        im_names = [os.path.join(cfg.image, cls, f) for
                f in os.listdir(os.path.join(cfg.image, cls)) if is_image(f)]
        train_paths.append(im_names)
        
    for cls_indice, cls in enumerate(unseen_cls):
        im_names = [os.path.join(cfg.image, cls, f) for
                f in os.listdir(os.path.join(cfg.image, cls)) if is_image(f)]
        test_paths.append(im_names)
    return seen_cls_indice, unseen_cls_indice, train_paths, test_paths


def preprocess_ps(cfg, all_cls_names):
    seen_cls = loadtxt(cfg.ps_seen_cls)
    seen_cls_indice = []

    unseen_cls = loadtxt(cfg.ps_unseen_cls)
    unseen_cls_indice = []

    for cls in unseen_cls:
        unseen_cls_indice.append(all_cls_names.index(cls))
    for cls in seen_cls:
        seen_cls_indice.append(all_cls_names.index(cls))

    train_seen_files = loadtxt(cfg.ps_train)
    train_seen_files_cls = [f[:f.find('/')] for f in train_seen_files]
    unseen_files = loadtxt(cfg.ps_test_unseen)
    unseen_files_cls = [f[:f.find('/')] for f in unseen_files]
    test_seen_files = loadtxt(cfg.ps_test_seen)
    test_seen_files_cls = [f[:f.find('/')] for f in test_seen_files]
        
    train_seen_paths = []
    test_unseen_paths = []
    test_seen_paths = []
    for cls_indice, cls in enumerate(unseen_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(unseen_files, unseen_files_cls) if cls == fcls]
        test_unseen_paths.append(im_names)
    for cls_indice, cls in enumerate(seen_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(train_seen_files, train_seen_files_cls) if cls == fcls]
        train_seen_paths.append(im_names)
    for cls_indice, cls in enumerate(seen_cls):
        im_names = [os.path.join(cfg.image, f) 
                for f, fcls in zip(test_seen_files, test_seen_files_cls) if cls == fcls]
        test_seen_paths.append(im_names)
    return seen_cls_indice, unseen_cls_indice, \
            train_seen_paths, test_seen_paths, test_unseen_paths


def parse_all(cfg):
    attr = prepare_attribute_matrix(cfg.attribute)
    attr = attr / np.linalg.norm(attr, axis=1, keepdims=True)
    class_name = prepare_cls_names(cfg.class_name)

    if cfg.split == 'SS':
        seen_cls_indice, unseen_cls_indice, train_paths, test_unseen_paths = \
            preprocess_ss(cfg, class_name)
        test_seen_paths = None # placeholder
    elif cfg.split == 'PS':
        seen_cls_indice, unseen_cls_indice, train_paths, test_seen_paths, test_unseen_paths = \
            preprocess_ps(cfg, class_name)
        ground_truth_indice = []
    else:
        raise NotImplementedError
    
    if cfg.test.setting == 'c':
        ground_truth_indice = []
        for i, _ in enumerate(unseen_cls_indice):
            ground_truth_indice.extend([i] * len(test_unseen_paths[i]))
    elif cfg.test.setting == 'g':
        assert cfg.split == 'PS'

        for i, _ in enumerate(seen_cls_indice):
            ground_truth_indice.extend([i] * len(test_seen_paths[i]))
        for i, _ in enumerate(unseen_cls_indice):
            ground_truth_indice.extend([i + len(seen_cls_indice)] * len(test_unseen_paths[i]))

    beta = ridge_regression(cfg.attribute, seen_cls_indice, unseen_cls_indice)
    return edict({
        "attr": attr, "seen_cls_indice": seen_cls_indice, "unseen_cls_indice": unseen_cls_indice,
        "train_paths": train_paths, "test_seen_paths": test_seen_paths, "test_unseen_paths": test_unseen_paths,
        "beta": beta, "ground_truth_indice": ground_truth_indice
    })


def predict_all(pth, cfg, parsing, device):

    model, image_size = load_model(pth, cfg, device)
    option = True
    if option:
        pred_attr, pred_latent, _, _ = inference(
            cfg, parsing.train_paths, parsing.seen_cls_indice, cfg.test.batch_size,
            model=model, device=device, image_size=image_size
        )
        seen_latent_prototype = np.zeros((len(parsing.seen_cls_indice), cfg.attr_dims), dtype=np.float32)
        cnt = 0
        for i, cls_indice in enumerate(parsing.seen_cls_indice):
            end = cnt + len(parsing.train_paths[i]) * croptimes[cfg.test.imload_mode]
            seen_latent_prototype[i, :] = np.mean(pred_latent[cnt: end, :], 0)
            cnt += len(parsing.train_paths[i]) * croptimes[cfg.test.imload_mode]
        unseen_latent_prototype = np.matmul(parsing.beta, seen_latent_prototype)
        latent_prototype = np.concatenate([
            seen_latent_prototype, unseen_latent_prototype], 0)

        np.save("latent_prototype.npy", latent_prototype)

    else:
        latent_prototype = np.load("latent_prototype.npy")

    if cfg.test.setting != 'g':
        pred_attr_unseen, pred_latent_unseen, featmap_unseen, pred_path_unseen = inference(
            cfg, parsing.test_unseen_paths, parsing.unseen_cls_indice, cfg.test.batch_size,
            model=model, device=device, image_size=image_size
        )
        return latent_prototype, pred_attr_unseen, pred_latent_unseen, featmap_unseen, pred_path_unseen

    else:

        pred_attr_unseen, pred_latent_unseen, featmap_unseen, pred_path_unseen = inference(
            cfg, parsing.test_unseen_paths, parsing.unseen_cls_indice, cfg.test.batch_size,
            model=model, device=device, image_size=image_size
        )

        pred_attr_seen, pred_latent_seen, featmap_seen, pred_path_seen = inference(
            cfg, parsing.test_seen_paths, parsing.seen_cls_indice, cfg.test.batch_size,
            model=model, device=device, image_size=image_size
        )
        pred_attr_last = np.concatenate([pred_attr_seen, pred_attr_unseen], 0)
        pred_latent_last = np.concatenate([pred_latent_seen, pred_latent_unseen], 0)
        featmaps = np.concatenate([featmap_seen, featmap_unseen], 0)
        pred_path = np.concatenate([pred_path_seen, pred_path_unseen], 0)

        return latent_prototype, pred_attr_last, pred_latent_last, featmaps, pred_path


def change_label(labels, std):
    label_o = np.zeros((labels.shape[0],labels.shape[1]), dtype=np.int)
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            label_o[i][j] = std[labels[i][j]]

    return label_o


def fix_gt_label(parsing,setting):

    cnt = 0
    # unseen
    if setting == 'c':

        label_ori = parsing.ground_truth_indice
        pathunseen = parsing.test_unseen_paths
        unseen_cls = parsing.unseen_cls_indice
        label = np.zeros(len(label_ori), dtype=np.int)
        for m in range(len(pathunseen)):
            for n in range(len(pathunseen[m])):
                label[cnt] = unseen_cls[m]
                cnt = cnt + 1

        return label

    # seen + unseen
    elif setting == 'g':
        label_ori = parsing.ground_truth_indice
        pathseen = parsing.test_seen_paths
        pathunseen = parsing.test_unseen_paths
        seen_cls = parsing.seen_cls_indice
        unseen_cls = parsing.unseen_cls_indice

        label = np.zeros(len(label_ori),dtype=np.int)

        for i in range(len(pathseen)):
            for j in range(len(pathseen[i])):
                label[cnt] = seen_cls[i]
                cnt = cnt + 1

        for m in range(len(pathunseen)):
            for n in range(len(pathunseen[m])):
                label[cnt] = unseen_cls[m]
                cnt = cnt + 1

        return label

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


def eval_c_seting(featmaps, pre_path, pred_labels, label_gt, gt_known, db_name, class_attribute):

    if gt_known:

        eval_c_gt_known(featmaps, pre_path, label_gt, db_name, class_attribute, setting='unseen')

    else:
        eval_c_gt_unknown(featmaps, pre_path, pred_labels, label_gt, db_name, class_attribute, setting='unseen')


def eval_g_seting(featmaps, pre_path, pred_labels_seen, pred_labels_unseen, label_gt, gt_known, db_name, class_attribute):

    if gt_known:

        eval_g_gt_known(featmaps, pre_path, label_gt, db_name, class_attribute)

    else:
        eval_g_gt_unknown(featmaps, pre_path, pred_labels_seen, pred_labels_unseen, label_gt, db_name, class_attribute)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, dest='cfg', required=True)
    parser.add_argument('--device', type=str, dest='device', default='1')
    parser.add_argument('--imload_mode', '-i', type=str, dest='imload_mode', default='')
    parser.add_argument('--checkpoint_base', '-c', default="./checkpoint")
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    update_datasets()

    cfg.test.imload_mode = args.imload_mode if args.imload_mode else cfg.test.imload_mode

    checkpoint_paths = parse_checkpoint_paths(cfg.test.epoch, osp.join(args.checkpoint_base, cfg.ckpt_name))
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device if args.device else cfg.gpu
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    parsing = parse_all(cfg)

    gt_known = False

    db_name = cfg.db_name
    if db_name == 'CUB':
        seen_num = 1764
        attr_path = "./loc_evaluation/cub/predicate-matrix-continuous.txt"
        class_attribute = read_class_attribute(attr_path)

    else:
        seen_num = 5882
        attr_path = "./loc_evaluation/awa2/predicate-matrix-continuous.txt"

        class_attribute = read_class_attribute(attr_path)

    for pth in checkpoint_paths:

        latent_prototype, pred_attr, pred_latent, featmaps, pre_path = predict_all(pth, cfg, parsing, device)

        print(pth)
        print("gt_known: ", gt_known)
        print("similarity_metric: ", cfg.hybrid.similarity_metric)


        if cfg.test.setting == 'c':

            pred_labels = hybrid_labeling(
                    pred_attr, pred_latent, parsing.attr[np.asarray(parsing.unseen_cls_indice), :],
                    latent_prototype[len(parsing.seen_cls_indice):, :], ensemble=cfg.hybrid.ensemble,
                    metric=cfg.hybrid.similarity_metric
            )
            pred_labels = crop_voting(cfg, [pred_labels])[0]
            pred_labels = change_label(pred_labels, parsing.unseen_cls_indice)
            label_gt = fix_gt_label(parsing, cfg.test.setting)

            eval_c_seting(featmaps, pre_path, pred_labels, label_gt, gt_known, db_name, class_attribute)

        elif cfg.test.setting == 'g':

            cls_indice = np.concatenate([parsing.seen_cls_indice, parsing.unseen_cls_indice], 0)

            pred_labels_seen = hybrid_labeling(
                    pred_attr[0:seen_num, :], pred_latent[0:seen_num, :],
                    parsing.attr[cls_indice, :],
                    latent_prototype[:, :], ensemble=cfg.hybrid.ensemble,
                    metric=cfg.hybrid.similarity_metric)

            pred_labels_seen = crop_voting(cfg, [pred_labels_seen])[0]

            pred_labels_unseen = hybrid_labeling(
                    pred_attr[seen_num:, :], pred_latent[seen_num:, :], parsing.attr[cls_indice, :],
                    latent_prototype[:, :], ensemble=cfg.hybrid.ensemble,
                    metric=cfg.hybrid.similarity_metric)

            pred_labels_unseen = crop_voting(cfg, [pred_labels_unseen])[0]

            pred_labels_seen = change_label(pred_labels_seen, cls_indice)

            pred_labels_unseen = change_label(pred_labels_unseen, cls_indice)

            label_gt = fix_gt_label(parsing, cfg.test.setting)

            eval_g_seting(featmaps, pre_path, pred_labels_seen, pred_labels_unseen, label_gt, gt_known, db_name, class_attribute)


if __name__ == "__main__":
    with torch.no_grad():
        main()

