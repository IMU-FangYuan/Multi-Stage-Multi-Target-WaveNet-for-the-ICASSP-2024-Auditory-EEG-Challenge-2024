# PYTORCH version of the vlaai original code.
import torch
import pdb

def pearson_correlation(y_true, y_pred, axis=1):

    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation.
    numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=False)

    std_true = torch.sum((y_true - y_true_mean)**2, dim=axis, keepdim=False)
    std_pred = torch.sum((y_pred - y_pred_mean)**2, dim=axis, keepdim=False)
    denominator = torch.sqrt(std_true * std_pred)
    
    pearsonR = torch.div(numerator, denominator + 1e-6)
    #p rint(pearsonR)

    assert torch.all(torch.lt(pearsonR, 1)) and torch.all(torch.gt(pearsonR, -1)), "Loss contains values outside the range of -1 to 1"

    return pearsonR


def pearson_torch(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in PyTorch.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: torch.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    torch.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=True,
    )
    std_true = torch.sum(torch.square(y_true - y_true_mean), dim=axis, keepdim=True)
    std_pred = torch.sum(torch.square(y_pred - y_pred_mean), dim=axis, keepdim=True)
    denominator = torch.sqrt(std_true * std_pred)

    # Compute the pearson correlation
    return torch.mean(torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator)), dim=-1)
    
    
    
def pearson_loss(y_true, y_pred, axis=1):
    return -pearson_correlation(y_true, y_pred, axis=axis)

def pearson_metric(y_true, y_pred, axis=1):
    return pearson_correlation(y_true, y_pred, axis=axis)
    
def l1_loss(y_true, y_pred, axis=1):
    l1_dist = torch.abs(y_true - y_pred)
    l1_loss = torch.mean(l1_dist, axis = axis, keepdim=False)
    return l1_loss
    
if __name__ == '__main__':
    est = torch.randn((1,640,10), dtype=torch.float32) 
    lab = torch.randn((1,640,10), dtype=torch.float32)
    #l1 = l1_loss(est,lab)
    p = pearson_correlation(est,lab) 
    #p = p+0.2*l1
    print('\n p:',p.mean())
    p1 = pearson_correlation(30*est,30*lab) 
    #p1 = pearson_torch(est,lab)
    #p1out = p1+ 0.2*l1.mean()
    print('\n p1:',p1.mean())