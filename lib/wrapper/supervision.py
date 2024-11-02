from lib.utils.registry import Registry
import torch
from lib.wrapper.preprocess import axis_angle_to_rot_6d
SUP = Registry('supervision')


@SUP.register()
def get_gt_smplx_params(wrapper, batch):
    smplx_params = batch['smplx_params'] = batch.get('smplx_params_oc', batch['smplx_params_hm'])
    batch['orient_6d'] = axis_angle_to_rot_6d(smplx_params['global_orient'])
    batch['body_pose_6d'] = axis_angle_to_rot_6d(smplx_params['body_pose'])
    

@SUP.register()
def get_object2pelvis(wrapper, batch):
    target_pelvis_xy = []
    for b in range(len(batch['meta'])):
        anchor = batch['meta'][b]['anchor']
        target_pelvis_xy.append(batch['pelvis'][b, anchor, :2])

    target_pelvis_xy = torch.stack(target_pelvis_xy, dim=0)
    batch['gt_anchor_pelvis_xy'] = target_pelvis_xy
    obj2pelivs_dist = (target_pelvis_xy[:, None] - batch['object_verts'][:, :, :2]).pow(2).sum(-1) # (B, N)
    obj2pelivs_dist = obj2pelivs_dist ** 2
    dmin = obj2pelivs_dist.min(-1)[0][:, None]
    dmax = obj2pelivs_dist.max(-1)[0][:, None]
    norm_dist = (obj2pelivs_dist - dmin) / (dmax - dmin)
    obj_verts_prob = 1 - norm_dist
    batch['gt_pointheat'] = obj_verts_prob
    # mask_9 = batch['gt_pointheat'][0] > 0.9
    # mask_8 = batch['gt_pointheat'][0] > 0.8
    # from lib.utils.vis3d_utils import make_vis3d
    # vis3d = make_vis3d(None, 'debug-heat')
    # vis3d.add_point_cloud(batch['object_verts'][0], name='all')
    # vis3d.add_point_cloud(batch['object_verts'][0][mask_9], name='9')
    # vis3d.add_point_cloud(batch['object_verts'][0][mask_8], name='8')
    pass

@SUP.register()
def get_gt_zero_vertices(wrapper, batch):
    B, S, _ = batch['smplx_params']['transl'].shape
    smplx_params = {
        'betas': batch['smplx_params']['betas'],
        'body_pose': batch['smplx_params']['body_pose'], 
    }
    smplx_params = {k: v.reshape(B * S, -1) for k, v in smplx_params.items()}
    body_opt = wrapper.smplx_model(**smplx_params, return_verts=True)
    body_vertices = body_opt.vertices
    batch.update({
        'gt_zero_verts': body_vertices.reshape(B, S, -1, 3),
    })

@SUP.register()
def get_gt_vertices(wrapper, batch):
    B, S, _ = batch['smplx_params']['transl'].shape
    smplx_params = {
        'betas': batch['smplx_params']['betas'],
        'transl': batch['smplx_params']['transl'],
        'global_orient': batch['smplx_params']['global_orient'],
        'body_pose': batch['smplx_params']['body_pose'], 
    }
    smplx_params = {k: v.reshape(B * S, -1) for k, v in smplx_params.items()}
    body_opt = wrapper.smplx_model(**smplx_params, return_verts=True)
    batch.update({
        'gt_verts': body_opt.vertices.reshape(B, S, -1, 3),
        'gt_joints': body_opt.joints.reshape(B, S, -1, 3),
    })