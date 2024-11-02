from lib.utils.net_utils import L1_loss, to_list
from lib.utils.registry import Registry
METRIC = Registry('metric')


@METRIC.register()
def recon_trans_metric(evaluator, batch):
    motion_mask = batch['motion_mask']
    # transl
    coord = evaluator.coord
    l1_loss = L1_loss(batch[f'smplx_params_{coord}']['transl'], batch['recon_transl']).mean(-1) * (~motion_mask)
    mean = l1_loss.sum(-1) / (~motion_mask).sum(-1)
    evaluator.update('R-Trans', to_list(mean))


@METRIC.register()
def recon_orient_metric(evaluator, batch):
    motion_mask = batch['motion_mask']
    # global orient
    coord = evaluator.coord
    l1_loss = L1_loss(batch[f'smplx_params_{coord}']['global_orient'], batch['recon_orient']).mean(-1) * (~motion_mask)
    mean = l1_loss.sum(-1) / (~motion_mask).sum(-1)
    evaluator.update('R-Orient', to_list(mean))
    

@METRIC.register()
def recon_localpose_metric(evaluator, batch):
    motion_mask = batch['motion_mask']
    coord = evaluator.coord
    # body pose
    l1_loss = L1_loss(batch[f'smplx_params_{coord}']['body_pose'], batch['recon_body_pose']).mean(-1) * (~motion_mask)
    mean = l1_loss.sum(-1) / (~motion_mask).sum(-1)
    evaluator.update('R-BodyPose', to_list(mean))