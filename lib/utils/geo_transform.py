import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from pytorch3d.transforms import *


def homo_points(points):
    """
    points: (B, N, 3)
    Returns: (B, N, 4), with 1 padded
    """
    return F.pad(points, [0, 1], value=1.)


def apply_T_on_points(points, T) -> torch.Tensor:
    """
    points: (B, N, 3) or (N, 3)
    T: (B, 4, 4) or (4, 4)
    Returns: (B, N, 3) or (N, 3)
    """
    points_batch = points[None] if len(points.shape) == 2 else points
    T_batch = T[None] if len(T.shape) == 2 else T
    points_h = homo_points(points_batch)
    points_T_h = torch.einsum('bki,bji->bjk', T_batch, points_h)
    if len(points.shape) == 2:
        return points_T_h[0, :, :3].contiguous()
    else:
        return points_T_h[..., :3].contiguous()


def transform_mat(R, t):
    """
    Args:
        R: Bx3x3 array of a batch of rotation matrices
        t: Bx3x(1) array of a batch of translation vectors
    Returns:
        T: Bx4x4 Transformation matrix    
    """
    # No padding left or right, only add an extra row
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)


def matrix_to_axis_angle(R):
    """ use pytorch3d so3_log_map
    Args:
        aa: (*, 3, 3)
    Returns:
        R: (*, 3)
    """
    return -quaternion_to_axis_angle(matrix_to_quaternion(R))


class ContinousRotReprDecoder(nn.Module):
    '''
    - this class encodes/decodes rotations with the 6D continuous representation
    - Zhou et al., On the continuity of rotation representations in neural networks
    - also used in the VPoser (see smplx)
    '''

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot


def axis_angle_to_rot_6d(x_r):
    """ axis-angle to rotation matrix (6D)

    Args:
        x_r: <..., D>

    Return:
        rotation matrix (6D), <..., D*2>
    """
    xr_shape = x_r.shape
    xr3 = x_r.reshape(-1, 3)
    xr_mat = ContinousRotReprDecoder.aa2matrot(xr3) # return [:,3,3]
    xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

    xr_repr = xr_repr.reshape([*xr_shape[:-1], xr_shape[-1]*2])
    return xr_repr


def rot_6d_to_axis_angle(x_r):
    """ rotation matrix (6D) to axis-angle

    Args:
        x_r: <..., D>
    
    Return:
        axis-angle, <..., D//2>
    """
    xr_shape = x_r.shape
    xr6 = x_r.reshape(-1, 6)
    xr_mat = ContinousRotReprDecoder.decode(xr6) # return [:,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]
    
    xr_aa = xr_aa.reshape([*xr_shape[:-1], xr_shape[-1]//2])
    return xr_aa