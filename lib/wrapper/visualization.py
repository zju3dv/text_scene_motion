from wis3d import Wis3D
from lib.utils.registry import Registry
import trimesh
VIS = Registry('visualization')


@VIS.register()
def vis_sample_two_stage(wrapper, batch, vis3d: Wis3D):
    scene_file = batch['meta'][0]['scene_ply_path']
    scene_mesh = trimesh.load(scene_file)
    t_to_scannet = batch['t_to_scannet']
    transl = batch['ksamples_transl'] + t_to_scannet
    B, SM, T, _ = transl.shape
    transl = transl.reshape(B*SM, T, -1)
    orient = batch['ksamples_orient'].reshape(B*SM, T, -1)
    body_pose = batch['ksamples_body_pose'].reshape(B*SM, T, -1)
    coord = wrapper.coord
    # gt_verts = process_smplx_ksamples(wrapper, batch[f'smplx_params_{coord}'], B*SM, T)
    pd_verts = process_smplx_ksamples(wrapper, batch[f'smplx_params_{coord}'], B*SM, T, transl=transl, orient=orient, body_pose=body_pose)
    pd_verts = pd_verts.reshape(B, SM, T, -1, 3)
    motion_mask = batch['motion_mask'][0]
    last_frame = (~motion_mask).sum() - 1
    scene_points, target = get_scene_target(batch, wrapper.coord)
    scene_points = scene_points + t_to_scannet
    for sm in range(SM):
        vis3d.set_scene_id(0)
        vis3d.add_point_cloud(batch['sample_stage1_transl'][0, sm, :last_frame] + t_to_scannet, name=f'sample-traj-{sm}')
        for t in range(T):
            if t == 0:
                vis3d.add_mesh(scene_mesh, name='scannet')
            vis3d.set_scene_id(t)
            if t > last_frame:
                break
            vis3d.add_point_cloud(pd_verts[0, sm, t], name=f'sample-body-{sm}')
            vis3d.add_point_cloud(scene_points, name='scene')
    pass


def get_scene_target(batch, coord):
    if 'pred_center' in batch:
        return batch[f'xyz_{coord}'][0], batch['pred_center']
    else:
        return batch[f'xyz_{coord}'][0], batch.get('object_center', None)

def process_smplx_ksamples(wrapper, smplx_params, K, S, transl=None, orient=None, body_pose=None):
    '''
    Input:
        K, S: shape
        transl, orient, body_pose: (K, S, C)
    Output:
        pd_verts (K, S, V, 3)
    '''
    betas = smplx_params['betas'][0].repeat((K, 1, 1))
    transl = smplx_params['transl'][0].repeat((K, 1, 1)) if transl is None else transl
    orient = smplx_params['global_orient'][0].repeat((K, 1, 1)) if orient is None else orient
    body_pose = smplx_params['body_pose'][0].repeat((K, 1, 1)) if body_pose is None else body_pose
    smplx_params = {
        'betas': betas,
        'transl': transl,
        'global_orient': orient,
        'body_pose': body_pose, 
    }
    smplx_params = {k: v.reshape(K * S, -1) for k, v in smplx_params.items()}
    body_opt = wrapper.smplx_model(**smplx_params, return_verts=True)
    pd_verts = body_opt.vertices.reshape(K, S, -1, 3)
    return pd_verts