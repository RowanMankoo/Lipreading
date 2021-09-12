"""
3 layer bidirectional GRU encoder is used to encode the source representation of the video, this is then passed to a 3 layered decoder with an attention mechanism attached to it. This produces vectorised scores for each class prediction.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import math

class Encoder(nn.Module):
    
    def __init__(self,
                input_dim,
                hid_dim,
                n_layers,
                dropout):
        super(Encoder, self).__init__()        
        self.GRU = nn.GRU(input_dim, hid_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc_hidden = nn.Linear(hid_dim*2, hid_dim)
        
    def forward(self, src):         
        
        # src = [batch_size, vid_seq_len, 512]
        
        enc_states, enc_final_hidden = self.GRU(src)       
        enc_final_hidden = torch.cat((enc_final_hidden[0][None,:,:],
                                      enc_final_hidden[1][None,:,:]), dim=2)
        enc_final_hidden = self.fc_hidden(enc_final_hidden)            
        
        # enc_states = [batch_size, vid_seq_len, 2*hid_dim]
        # enc_final_hidden = [1, batch_size, hid_dim]
    
        return enc_states, enc_final_hidden
    
    
class Decoder(nn.Module):
    
    """
    Decoder outputs only one character prediction at a time with every forward call
    
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                 n_layers,
                 dropout):
        super(Decoder, self).__init__()
        
        self.GRU = nn.GRU(hid_dim*2+input_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)       
        self.energy = nn.Linear(hid_dim*3, 1)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        
        
    def forward(self, inp, enc_states, hidden, max_len=100):
        
        # inp = [batch_size, 1, 42]
        # enc_states = [batch_size, vid_seq_len, hid_dim*2]
        # hidden = [n_layers, batch_size, hid_dim]
        
        # reshape inputs 
        batch_size = enc_states.shape[0]
        vid_seq_len = enc_states.shape[1]
        hidden_ = hidden[0,:,:]
        hidden_ = hidden_.repeat(vid_seq_len,1,1)
        enc_states_ = enc_states.transpose(0,1)
        
        energy_inp = torch.cat((hidden_, enc_states_),dim=2)  # energy_inp = [vid_seq_len, batch_size, 3*hid_dim]
       
        # calculate attention scores
        energy = torch.tanh(self.energy(energy_inp)) 
        attention = F.softmax(energy, dim=0)                  # attention = [vid_seq_len, batch_size, 1]
        
        # create linear combination of attention scores and encoder states
        attention = attention.permute(1,2,0)                  # attention = [batch_size, 1, vid_seq_len]
        context = torch.bmm(attention, enc_states)            # context = [batch_size, 1, hid_dim*2]        
        
        concat_inp = torch.cat((context,inp), dim=2)  
        dec_out, next_hidden = self.GRU(concat_inp, hidden)
        preds = self.fc_out(dec_out).squeeze(dim=1)
        
        # preds = [n_batches, 42]
        # next_hidden = [n_layers, batch_size, hid_dim]
        
        return preds, next_hidden
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, num_dec_layers, device, hid_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_dec_layers = num_dec_layers
        self.hid_dim = hid_dim
        
    def forward(self, src, trgt, teacher_force_ratio, max_len=100):
        
        # src = [batch_size, vid_seq_len, 512]
        # trgt = [text_seq_len, batch_size]
        
        batch_size = src.shape[0]
        vid_seq_len = src.shape[1]
        if trgt == None: 
            trgt_len = 100
            teacher_force_ratio = 0
            
        else:
            trgt_len = trgt.shape[0]
            
            
        outputs = torch.zeros(batch_size, trgt_len, 42).to(self.device)
        # encoder stage
        enc_states, hidden = self.encoder(src)  
        if self.num_dec_layers>1:
            pad_hidden = torch.zeros(self.num_dec_layers-1,batch_size,self.hid_dim).to(self.device)
            hidden = torch.cat((hidden,pad_hidden),0)
        
        # initalise "<SOS>" token
        x = torch.zeros(batch_size,1,42).to(self.device)   
        x[:,:,-1]=1
        outputs[:,0,:] = x.squeeze()                          # automatically add "<SOS>" to first of outputs
                
        # generate predictions one by one
        for t in range(1, trgt_len):

            # decoder stage
            preds, hidden = self.decoder(x, enc_states, hidden)
            outputs[:,t,:] = preds.squeeze()
            
            # apply teacher force ratio
            preds = preds.argmax(1) 
            preds = F.one_hot(preds.squeeze(), num_classes=42).reshape(-1,42) 
            preds = preds[:,None,:].type('torch.FloatTensor')
            if random.uniform(0,1) < teacher_force_ratio:
                x = F.one_hot(trgt[t].to(torch.int64), num_classes=42)[:,None,:].type('torch.FloatTensor')
            else:
                x = preds
            # preds/acc_trgt = [batch_size, 1, 42]
            x = x.to(self.device)
            
        # outputs = [batch_size, text_seq_len, 42]
        
        return outputs

