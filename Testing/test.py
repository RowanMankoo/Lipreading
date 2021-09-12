"""
This .py file iterates over the whole training set to calculate the loss metrics CER/WER

batch_size(int): Number of batches per propagation
pad_indx(int): The numerical value of the '<PAD>' index, which is 0 in the case of our vocabulary

file_directory(str): location of video and text files (for test data)
index_directory(str): Location of index names for text and video files of test data
model_save_path(str): Location to store and save model.pth file
"""
from Data import *
from final_model import *
from utils import *
import torch.optim as optim
from jiwer import wer
import matplotlib.pyplot as plt
from pynvml import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.cuda.empty_cache()
p = {'batch_size':1,  
     'pad_indx':0,
     'file_directory':"/data/fast/rxm064/temp/mvlrs_v1/main",
     'index_directory':"/data/fast/rxm064/temp/mvlrs_v1/train.txt",
     'model_save_path':'/data/fast/rxm064/temp/model.pth'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# data loader
dataset = LRS2Dataset(file_directory=p['file_directory'],
                            index_directory=p['index_directory'])
data_loader = DataLoader(dataset=dataset, 
                         batch_size=p['batch_size'],
                         shuffle=True, 
                         collate_fn=MyCollate())
n_total_steps = len(data_loader)

# load model
model = Lipreading(device=device)
model.to(device)
checkpoint = torch.load(p['model_save_path'])
model.load_state_dict(checkpoint['model_state'])



loss_, val_loss_ = 0, 0
wer__, val_wer__ = 0, 0
cer__, val_cer__ = 0, 0
running_wer, val_running_wer = 0, 0
running_cer, val_running_cer = 0, 0
running_loss, val_running_loss = 0, 0

average_wer_ = 0
average_cer = 0
average_loss = 0
for text,vid in enumerate(data_loader):
    # send to correct device
    text = text.to(device)
    vid = vid.to(device)
    
    # forward pass
    preds = model(text=None, vid=vid, teacher_force_ratio=p['teacher_force_ratio'])
    text = itos(text.transpose(0,1))
    preds = itos(terminate_seq_early(preds))
    
    # calculate metrics
    wer_ = wer(text,preds)
    cer_ = CER(preds, text)
    average_wer += wer_/n_total_steps
    average_cer += cer_/n_total_steps
    
print(f'The Average WER is: {average_wer}')    
print(f'The Average CER is: {average_cer}')    