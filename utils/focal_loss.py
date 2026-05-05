import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss实现，用于解决类别不平衡问题
    
    Focal Loss = -α(1-pt)^γ * log(pt)
    其中：
    - pt是模型预测正确类别的概率
    - α是平衡因子，用于平衡正负样本
    - γ是聚焦参数，用于降低易分类样本的权重
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): 平衡因子，用于平衡正负样本的权重
            gamma (float): 聚焦参数，用于降低易分类样本的权重
            reduction (str): 损失的归约方式，'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型的原始输出 (logits)，形状为 [batch_size, 1] 或 [batch_size]
            targets: 真实标签，形状为 [batch_size]，值为0或1
        
        Returns:
            focal_loss: 计算得到的focal loss
        """
        # 确保inputs和targets的形状一致
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)  # pt = exp(-BCE) = p if y=1, 1-p if y=0
        
        # 计算alpha权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"