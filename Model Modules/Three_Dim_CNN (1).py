import torch.nn as nn
import torch.nn.functional as F

class Three_dim_CNN(nn.Module):
    
    """
    First stage of model, applies a 3D CNN to input videos to generate a lower dimensional represnetation of the
    video whilst preserving the number of frames present
    """
    def __init__(self):
        super(Three_dim_CNN,self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(5, 4, 4),
                          stride=(2, 1, 1), padding=(2, 3, 3), bias=False)
        self.bn = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 0, 0))
    
    def forward(self, inp):
        inp = inp[:,None,:,:,:]     # inp = [batch_size, 1, original_vid_seq_len, 112, 112]
        out = self.conv1(inp)
        out = F.relu(self.bn(out))
        out = self.pool(out)        # out = [batch_size, 64, vid_seq_len, 57, 57]
        
        return out   

