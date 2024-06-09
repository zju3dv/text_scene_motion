# Author: zhshen0917@gmail.com, Date: 23-04-30

import torch

@torch.no_grad()
def compute_sample_error(sample, smplx_verts, smplx_faces, metric_dict, id2scene, method, K, vis=False, vis3d=None):
    """ values in gt and pred dict are all tensor, the outputs are also tensors """
    T = sample['pred_params']['trans'].shape[1]
    scene_id = sample['scene_id']
    name = 'gt' if K == 1 else 'sample'
    scene_dict = id2scene[scene_id]
    xyz_sn = torch.FloatTensor(scene_dict['xyz_scan'])
    
    # dist_anchor_to_tgt
    if 'dist_anchor_to_tgt' in metric_dict:
        smplx_verts_anchor = smplx_verts[:, sample['anchor_index']]  # (K, V, 3) in pred coord
        if method == 'humanise':
            object_points = sample['tgt_points'][None].repeat(K, 1, 1).cuda()  # (K, N, 3)
        else:
            if K == 1:
                gt_object_points = xyz_sn[sample['gt_params']['tgt_mask']] - sample['gt_params']['t_to_scannet']
            else:
                gt_object_points = xyz_sn[sample['gt_params']['tgt_mask']] - sample['pred_params']['t_to_scannet']
            
            object_points = gt_object_points[None].repeat(K, 1, 1).cuda()
        
        object_to_human_sdf, _ = smplx_signed_distance(object_points, smplx_verts_anchor, smplx_faces) # (K, N)        

        # dist_anchor_to_tgt = min(object_to_human_sdf.max().cpu().item(), 0.)
        # I found that the above line is not correct,
        # According to the paper, the following lines take the shortest non-negative distance 
        #   and average over all samples
        object_to_human_sdf = (object_to_human_sdf).clamp(min=0.)
        dist_anchor_to_tgt = object_to_human_sdf.min(-1)[0].mean().cpu()

        metric_dict['dist_anchor_to_tgt'].append(dist_anchor_to_tgt)

    if 'dist_anchor_to_pred_tgt' in metric_dict:
        smplx_verts_anchor = smplx_verts[:, sample['anchor_index']]  # (K, V, 3)
        if method == 'humanise':
            object_points = sample['pred_params']['tgt_center'][None, None].repeat(K, 1, 1).cuda() # (K, 1, 3)
        else:
            object_points = xyz_sn[sample['gt_params']['tgt_mask']][None].repeat(K, 1, 1).cuda() - sample['pred_params']['t_to_scannet']
        
        object_to_human_sdf, _ = smplx_signed_distance(object_points, smplx_verts_anchor, smplx_faces) # (K, N)   
        
        object_to_human_sdf = (object_to_human_sdf).clamp(min=0.)
        dist_anchor_to_tgt = object_to_human_sdf.min(-1)[0].mean().cpu()

        metric_dict['dist_anchor_to_pred_tgt'].append(dist_anchor_to_tgt)


# =====  Evaluation metric util functions ===== #

# the distance from anchor pose to target object
def smplx_signed_distance(object_points, smplx_vertices, smplx_face):
    """ Compute signed distance between query points and mesh vertices.
        If the query point is outside the mesh, the distance is POSITIVE. 
        This function is different from the one in HUMANISE.
    
    Args:
        object_points: (B, O, 3) query points in the mesh.
        smplx_vertices: (B, H, 3) mesh vertices.
        smplx_face: (F, 3) mesh faces.
    
    Return:
        signed_distance_to_human: (B, O) signed distance to human vertex on each object vertex
        closest_human_points: (B, O, 3) closest human vertex to each object vertex
        signed_distance_to_obj: (B, H) signed distance to object vertex on each human vertex
        closest_obj_points: (B, H, 3) closest object vertex to each human vertex
    """
    # compute vertex normals
    smplx_face_vertices = smplx_vertices[:, smplx_face]
    e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
    e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]
    e1 = e1 / torch.norm(e1, dim=-1, p=2).unsqueeze(-1)
    e2 = e2 / torch.norm(e2, dim=-1, p=2).unsqueeze(-1)
    smplx_face_normal = torch.cross(e1, e2)     # (B, F, 3)

    # compute vertex normal
    smplx_vertex_normals = torch.zeros_like(smplx_vertices).float()
    smplx_vertex_normals.index_add_(1, smplx_face[:,0], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,1], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,2], smplx_face_normal)
    smplx_vertex_normals = smplx_vertex_normals / torch.norm(smplx_vertex_normals, dim=-1, p=2).unsqueeze(-1)

    # compute paired distance of each query point to each face of the mesh
    pairwise_distance = torch.norm(object_points.unsqueeze(2) - smplx_vertices.unsqueeze(1), dim=-1, p=2)    # (B, O, H)
    
    # find the closest face for each query point
    distance_to_human, closest_human_points_idx = pairwise_distance.min(dim=2)  # (B, O)
    closest_human_point = smplx_vertices.gather(1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, O, 3)

    # NOTE: vec{query->human} * human_normal > 0 means query is inside the human
    query_to_surface = closest_human_point - object_points  
    query_to_surface = query_to_surface / torch.norm(query_to_surface, dim=-1, p=2).unsqueeze(-1)
    closest_vertex_normals = smplx_vertex_normals.gather(1, closest_human_points_idx.unsqueeze(-1).repeat(1, 1, 3))
    same_direction = torch.sum(query_to_surface * closest_vertex_normals, dim=-1)
    signed_distance_to_human = same_direction.sign() * distance_to_human    # (B, O)
    signed_distance_to_human = -signed_distance_to_human

    return signed_distance_to_human, closest_human_point


@torch.no_grad()
def get_sample_motions(smplx_model, sample, device, add_more_to_sample=False):
    smplx_model = smplx_model.to(device)
    
    # gt_params
    params_ = sample['gt_params']    
    T = params_['trans'].size(0)
    smplx_params = {
        'betas': params_['betas'][None],
        'global_orient': params_['orient'],  # [T, 3]
        'transl':  params_['trans'],
        'body_pose':  params_['pose_body']}
    smplx_params = {k: v.to(device) for k, v in smplx_params.items()}
    smplx_output = smplx_model(**smplx_params)
    gt_motions = smplx_output.vertices.reshape(1, T, -1, 3) # (1, T, V, 3)

    # pred_params
    params_ = sample['pred_params']    
    K = params_['trans'].size(0)
    assert T == params_['trans'].size(1)
    smplx_params = {
        'betas': params_['betas'][None],
        'global_orient': params_['orient'].reshape(K*T, 3),  # [KT, 3]
        'transl':  params_['trans'].reshape(K*T, 3),
        'body_pose':  params_['pose_body'].reshape(K*T, -1)}
    smplx_params = {k: v.to(device) for k, v in smplx_params.items()}
    smplx_output = smplx_model(**smplx_params)
    pred_k_motions = smplx_output.vertices.reshape(K, T, -1, 3) # (K, T, V, 3)

    if add_more_to_sample:
        sample["pred_k_joints"] = smplx_output.joints.reshape(K, T, -1, 3).detach().cpu()  # (K, T, 24, 3)

    return gt_motions, pred_k_motions