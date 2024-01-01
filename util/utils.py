import torch
import numpy as np
import os

import math

        

def save_checkpoint(model, optimizer, learning_rate, epoch, filepath):
    print(f"Saving model and optimizer state at iteration {epoch} to {filepath}")
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    torch.save({'epoch': epoch,
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{epoch}')
    
def save_checkpoint_best(model, optimizer, learning_rate, epoch, filepath):
    print(f"Saving model and optimizer state at iteration {epoch} to {filepath}")
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    torch.save({'epoch': epoch,
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_best')
    
