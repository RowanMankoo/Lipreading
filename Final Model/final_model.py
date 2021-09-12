from Three_Dim_CNN import *
from resnet import *
from Seq_to_Seq import *
from experiment import *

class Lipreading(nn.Module):
    
    """
    Final lipreading model, outputs vectorised scores for each class prediction
    """
    def __init__(self, device):
        super(Lipreading, self).__init__()
        self.Three_dim_CNN = Three_dim_CNN()
        self.res = resnet_()
        self.enc = Encoder(input_dim=512, hid_dim=1024, n_layers=3, dropout=0.1)
        self.dec = Decoder(input_dim=42, hid_dim=1024, n_layers=3, output_dim=42, dropout=0.1)
        self.seqtoseq = Seq2Seq(encoder=self.enc, decoder=self.dec, device=device, num_dec_layers=3, hid_dim=1024)
        
    def forward(self, text, vid, teacher_force_ratio):
        
        # text = [text_seq_len, batch_size]
        # vid = [batch_size, vid_seq_len, 112, 112]
        batch_size = vid.shape[0]
        
        out = self.Three_dim_CNN(vid)                        # out = [batch_size, 64, vid_seq_len, 57, 57]    
        out = out.transpose(1,2)  
        out = out.reshape(-1, 64, out.size(3), out.size(4))  # out = [batch_size*vid_seq_len, 64, 57, 57]
        
        out = self.res(out)                                  # out = [batch_size, vid_seq_len, 512]
        out = out.reshape(batch_size, -1, 512)               # out = [batch_size, vid_seq_len, 512]
        
        out = self.seqtoseq(out, text, teacher_force_ratio)  # out = [batch_size, text_seq_len, 42]
        
        return out

