import torch
import torch.nn.functional as F

def edge_loss(input, target):
    # input: [B,1,H,W] 预测（0~1），target: [B,1,H,W] 真实 mask（0 或 1）
    # 先定义 Sobel 卷积
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      dtype=torch.float32, device=input.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                      dtype=torch.float32, device=input.device).view(1,1,3,3)
    # 计算梯度幅值
    gx_pred = F.conv2d(input, kx, padding=1)
    gy_pred = F.conv2d(input, ky, padding=1)
    edge_pred = torch.sqrt(gx_pred**2 + gy_pred**2)

    gx_gt = F.conv2d(target.float(), kx, padding=1)
    gy_gt = F.conv2d(target.float(), ky, padding=1)
    edge_gt = torch.sqrt(gx_gt**2 + gy_gt**2)

    # L1 对齐
    return F.l1_loss(edge_pred, edge_gt)
