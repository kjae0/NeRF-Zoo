import torch
import torch.nn as nn


class BaseEngine:
    def __init__():
        pass
    
    def train_one_epoch(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError    


