import sys
sys.path.append("..") 

import torch
import torch.nn as nn

class MolOptDecoder(nn.Module):
    def __init__(self, args):
        super(MolOptDecoder, self).__init__()
        self.args = args

    def add_delta(self, x_embedding, xy_delta):
        """Get the new embedding from the old one, plus the error term""" 
        return (x_embedding + xy_delta)