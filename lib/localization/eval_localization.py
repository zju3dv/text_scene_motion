import torch

def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )

def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )

def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union

def get_bbx(points):
    points = torch.FloatTensor(points)
    return torch.cat((points.min(0)[0], points.max(0)[0]))

def evaluate_pd_bbox_by_span(pred_points, gt_points, mode):
    # Measure IoU>threshold
    if mode == 'gt':
        gt_bbx_max = gt_points.max(0) + 0.05
        gt_bbx_min = gt_points.min(0) - 0.05
        for pred_point in pred_points:
            if (pred_point < gt_bbx_max).all() and (pred_point > gt_bbx_min).all():
                found = True
            else:
                found = False
    else:
        # IoU
        ious, _ = _iou3d_par(
            get_bbx(gt_points)[None],
            get_bbx(pred_points)[None]
        ) 
        found = ious > 0.25
    return found
