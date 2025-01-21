import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 传统动量法更新方式
def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update

def momentum_update_m2(old_value, new_value, momentum, debug=False):
    updated_value = momentum.unsqueeze(-1) * old_value + (1 - momentum.unsqueeze(-1)) * new_value
    return updated_value

# 自适应动量法更新方式
def adaptive_momentum_update(old_value, new_value, momentum, update_count):
    adaptive_momentum = momentum * (1 - math.exp(-update_count / 1000))
    return adaptive_momentum * old_value + (1 - adaptive_momentum) * new_value

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

# 投影头进行降维和投影，常用于对比学习或语义分割等任务
# 当输入和输出特征完全一致时，其重要作用是作为一个中间学习层，用于特征的非线性变换
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))

# new modified MLP module
class MLP(nn.Module):
    def __init__(self, dim_in,proj_dim=256) -> None:
        super(MLP,self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_in,dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in,proj_dim)
        )
    
    def forward(self,x):
        return l2_normalize(self.mlp(x))