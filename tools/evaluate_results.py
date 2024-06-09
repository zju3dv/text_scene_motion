import os
import argparse
from pathlib import Path
import numpy as np
import torch
import pickle
from tqdm import tqdm

from lib.config import make_cfg
from lib.utils.comm import seed_everything
from lib.utils.smplx_utils import make_smplx
from lib.utils.geo_transform import matrix_to_axis_angle, axis_angle_to_matrix
from lib.utils.evaluate.fid_utils import *
from lib.utils.evaluate.eval_utils import compute_sample_error, get_sample_motions

# constants
USE_ALIGN_Y = True
align_yup = torch.FloatTensor([1,0,0,0,0,1,1,0,0]).reshape(3,3)
num_classes = 4
action2label = {'walk': 0, 'sit': 1, 'standup': 2, 'lie': 3, 'stand up': 2}


def get_x(params, key):
    orient, pose_body = params['orient'], params['pose_body']
    # align y up
    if USE_ALIGN_Y:
        orient = matrix_to_axis_angle((align_yup @ axis_angle_to_matrix(orient)))

    x = torch.cat([orient, pose_body], dim=-1).to(device)
    if key == 'gt':
        K, T = 1, x.size(0)
    else:
        K, T = x.shape[:2]
    # add zero hand wrists, convert to 6d
    x = F.pad(x, (0, 6), mode='constant', value=0)  # add zero hand wrist
    x = x.reshape(K, T, 24, 3) # (K, T, 24, 3)

    if T > 120:
        orient = orient[:, : 120]
        pose_body = pose_body[:, : 120]
    else:
        padding = torch.tile(x[:, [-1]], (1, 120 - T, 1, 1))
        x = torch.cat([x, padding], dim=1)
    
    x = convert_x_to_rot6d(x).permute(0, 2, 3, 1)  # (K, T, 24, 6) -> (K, 24, 6, T)
    return x

def compute_acc_fid_div_mm(samples, action='all'):
    feats_gt, feats_pred_k, labels, gt_labels, pred_labels_k = [], [], [], [], []
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        gt_x = get_x(sample['gt_params'], 'gt')
        pred_x = get_x(sample['pred_params'], 'pred')

        gt_feat, gt_label = compute_features_labels(STGCN, gt_x)
        pred_feat, pred_label = compute_features_labels(STGCN, pred_x)        
        feats_gt.append(gt_feat)  # (K, F)
        feats_pred_k.append(pred_feat)  # (K, F)
        gt_labels.append(gt_label)
        pred_labels_k.append(pred_label)

        if 'meta' in sample:
            labels.append(action2label[sample['meta']['action']])
        else:
            if action == 'all':
                if i < 346:
                    labels.append(action2label['lie'])
                elif i < 346+877:
                    labels.append(action2label['sit'])
                elif i < 346+877+583:
                    labels.append(action2label['stand up'])
                else:
                    labels.append(action2label['walk'])
            else:
                labels.append(action2label[action])
                
    feats_gt = torch.cat(feats_gt, dim=0)  # (B, F)
    feats_pred_k = torch.stack(feats_pred_k, dim=1)  # (K, B, F)
    labels = torch.LongTensor(labels) # (B,)
    gt_labels = torch.LongTensor(gt_labels) # (B,)
    pred_labels_k = torch.cat(pred_labels_k, dim=0)  # (B, K)

    # gt accuarcy
    confusion = torch.zeros(num_classes, num_classes)
    for b in range(gt_labels.size(0)):
        confusion[gt_labels[b], labels[b]] += 1
    gt_accuracy = torch.trace(confusion) / torch.sum(confusion)
    # pred accuarcy
    confusion = torch.zeros(num_classes, num_classes)
    for b in range(pred_labels_k.size(0)):
        for k in range(pred_labels_k.size(1)):
            confusion[pred_labels_k[b, k], labels[b]] += 1
    pd_accuracy = torch.trace(confusion) / torch.sum(confusion)

    # gt fid
    gt_stats = calculate_activation_statistics(feats_gt)
    gt_fid = calculate_fid(gt_stats, gt_stats)
    # pred fid
    fids = []
    for k in range(feats_pred_k.size(0)):
        pred_stats = calculate_activation_statistics(feats_pred_k[k])
        fid = calculate_fid(pred_stats, gt_stats)
        fids.append(fid)
    pd_fid = np.mean(fids)

    # gt div, mm
    gt_div, gt_mm = calculate_diversity_multimodality(feats_gt, labels, num_classes, seed=42)
    # pred div, mm
    pred_div, pred_mm = [], []
    for k in range(10):
        div, mm = calculate_diversity_multimodality(feats_pred_k[k], labels, num_classes, seed=42)
        pred_div.append(div)
        pred_mm.append(mm)
    pd_div = np.mean(pred_div)
    pd_mm = np.mean(pred_mm)
    print(f'acc: {gt_accuracy:.3f}; pred: {pd_accuracy:.3f} div: {gt_div:.3f}; pred: {pd_div:.3f}')
    print(f'mm : {gt_mm:.3f}; pred: {pd_mm:.3f} fid: {gt_fid:.3f}; pred: {pd_fid:.3f}')


def compute_dist(samples, method):
    # load data
    pkl_file = f'data/humanise_preprocess/all_test_data.pkl'
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            pdata = pickle.load(f)
        id2scene = pdata['id2scene']
        
    gt_metric_dict = {'dist_anchor_to_tgt': [],}
    metric_dict = {'dist_anchor_to_tgt': [],}
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        gt_motions, pred_k_motions = get_sample_motions(smplx_model, sample, device, add_more_to_sample=True)  # (1, T, V, 3), (K, T, V, 3)
        compute_sample_error(sample, gt_motions, smplx_face, gt_metric_dict, id2scene, method=method, K=1)
        compute_sample_error(sample, pred_k_motions, smplx_face, metric_dict, id2scene, method=method, K=10)
        
    gt_metric_dict = {k:np.array(v) for k, v in gt_metric_dict.items()}
    metric_dict = {k:np.array(v) for k, v in metric_dict.items()}
    # Print results
    gt_dist = gt_metric_dict['dist_anchor_to_tgt'].mean()
    pd_dist = metric_dict['dist_anchor_to_tgt'].mean()
    print(f'goal dist: {gt_dist:.3f}; pred: {pd_dist:.3f}')


def main():
    method = cfg.method
    action = cfg.action
    
    output_dir = Path(cfg.record_dir)
    folder_names = cfg.get(method).get(action)

    output_pkl_fns = [output_dir / folder_name / 'sample.pkl' for folder_name in folder_names]
    samples = []
    for output_pkl_fn in output_pkl_fns:
        samples.extend(pickle.load(output_pkl_fn.open('rb')))
    print(f"method: {method}; folder_names: {folder_names}; total {len(samples)} samples")

    if cfg.eval_t2m:
        compute_acc_fid_div_mm(samples, action=action)
    if cfg.eval_dist:
        compute_dist(samples, method=method)



if __name__ == '__main__':
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c", default="configs/test/evaluate.yaml")
    parser.add_argument("--is_test", action="store_true", default=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    # set device, load smplx models
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device)
    device = torch.device('cuda')
    smplx_model = make_smplx('humanise').to(device)
    smplx_face = torch.from_numpy(smplx_model.bm.faces.astype(np.int64)).cuda()

    # load model
    STGCN = initialize_model(device, cfg.stgcn)
    main()