import torch
import torch.nn as nn

class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss:
    Computes a weighted mean squared error where the weight (focal_factor)
    is defined as |error|^alpha. This emphasizes examples with larger errors.
    
    Formula:
        loss = |error|^alpha * (error^2)
    where error = input - target, and alpha controls the focus on large errors.
    
    Parameters:
        alpha (float): Exponent factor to increase the penalty for larger errors.
        reduction (str): Specifies the reduction to apply to the output:
                         'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha=1.0, reduction='mean'):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # Compute the error between predictions and targets
        error = input - target
        
        # Compute the standard Mean Squared Error
        mse = error ** 2
        
        # Compute the focal factor to emphasize larger errors
        focal_factor = torch.abs(error) ** self.alpha
        
        # Combine the focal factor with the MSE
        loss = focal_factor * mse
        
        # Apply the specified reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Testing the FocalMSELoss
if __name__ == "__main__":
    criterion = FocalMSELoss(alpha=2.0, reduction='mean')
    inputs = torch.tensor([0.2, 0.7, 1.5, 2.0], dtype=torch.float32)
    targets = torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float32)
    loss_val = criterion(inputs, targets)
    print("Loss =", loss_val.item())
