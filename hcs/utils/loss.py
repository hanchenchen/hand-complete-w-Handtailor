import torch
# from chamfer_distance import ChamferDistance
from chamferdist import ChamferDistance


def chamfer_loss(src, dst):
    """
    Pointset to pointset matching loss
    :param src: Source point set, B x N x 3
    :param dst: Target mesh, B x 778 x 3
    :return chamfer_loss
    """
    chamfer_dist = ChamferDistance()
    # dist1, dist2 = chamfer_dist(src, dst)
    return chamfer_dist(src.cpu(), dst.cpu())
    dist1, dist2, _, _ = chamfer_dist(src.cpu(), dst.cpu())
    loss = torch.mean(dist1) + torch.mean(dist2)
    # loss = torch.mean(dist2)
    # loss = torch.mean(dist1)
    return loss
