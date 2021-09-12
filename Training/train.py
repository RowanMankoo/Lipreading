"""
This .py file represents a normal training loop. Hyperparameters of the loop can be set in the dictionary p. Note that due to memory constraints of the GPU Gradient accumulation is being utilised in the code, the corresponding hyperparameter controlling this is p['accumulation_steps']. Below is a description of all hyperparameters used throughout the training.

learning rate(float): The learning rate used 
batch_size(int): Number of batches per propagation
accumulation_steps(int): For how many steps to accumulate gradients for 
n_words(int): Number of words in current curriculum 
pad_indx(int): The numerical value of the '<PAD>' index, which is 0 in the case of our vocabulary
load_model(bool): States whether to laod a pretrained model or not
load_losses(bool): States whether to load the stored losse, in case of training on curriculum getting interupted
n_steps(int): Number of steps to take during current curriculum
teacher_force_ratio(float): Value between [0,1]
save_freq(int): After how many steps to checkpoint model and losses

file_directory(str): location of video and text files (for main training data)
index_directory(str): Location of index names for text and video files of training data
val_index_directory(str): Location of index names for text anf video files of validation data
train_losses_file(str): Location to store and save loss/metrics file for train data
val_losses_file(str):Location to store and save loss/metrics file for validation data
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
p = {'learning_rate':,
     'batch_size':,
     'accumulation_steps':,
     'pad_indx':0,
     'load_model':,
     'load_losses':,
     'n_steps':,
     'teacher_force_ratio':,
     'save_freq':,
     'file_directory':,
     'index_directory':,
     'val_index_directory':,
     'train_losses_file':,
     'val_losses_file':,
     'model_save_path':}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# data loaders
dataset = LRS2Dataset(file_directory=p['file_directory'],
                            index_directory=p['index_directory'])
data_loader = DataLoader(dataset=dataset, 
                         batch_size=p['batch_size'],
                         shuffle=True, 
                         collate_fn=MyCollate())
n_total_steps = len(data_loader)
val_dataset = LRS2Dataset(file_directory=p['file_directory'],
                         index_directory=p['val_index_directory'])
val_data_loader = DataLoader(dataset=val_dataset, 
                         batch_size=p['batch_size'],
                         shuffle=True, 
                         collate_fn=MyCollate())

# define model together with loss function and optimizer to be used
model = Lipreading(device=device)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=p['pad_indx'])
optimizer = optim.Adam(model.parameters(), lr=p['learning_rate'])
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.4, min_lr=0.00000005,
                              patience=1200, verbose=True, threshold=0.01)


# load model
if p['load_model']:
    checkpoint = torch.load(p['model_save_path'])
    model.load_state_dict(checkpoint['model_state'])
    #optimizer.load_state_dict(checkpoint['optim_state'])
# unfreze resnet-18
for param in model.parameters():
    param.requires_grad=True
model.train()
    
# load losses
training_loss, training_CER, training_WER = [], [], []
validation_loss, validation_CER, validation_WER = [], [], []
if p['load_losses']:
    losses = torch.load(p['train_losses_file'])  
    training_loss = losses['training_loss']
    training_CER = losses['training_CER']
    training_WER = losses['training_WER']
    
    losses = torch.load(p['val_losses_file'])  
    validation_loss = losses['val_training_loss']
    validation_CER = losses['val_training_CER']
    validation_WER = losses['val_training_WER']
loss_, val_loss_ = 0, 0
wer__, val_wer__ = 0, 0
cer__, val_cer__ = 0, 0
running_wer, val_running_wer = 0, 0
running_cer, val_running_cer = 0, 0
running_loss, val_running_loss = 0, 0

for i in range(p['n_steps']):
    
    text, vid = next(iter(data_loader))
    
    # set number of iterations
    if i >= p['n_steps']:
        break

    # send to correct device
    text = text.to(device)
    vid = vid.to(device)

    # train forward pass
    preds = model(text=None, vid=vid, teacher_force_ratio=p['teacher_force_ratio'])
    preds, text = pad_outputs(preds,text, device=device)   # pad outputs to match dimensions
    preds_ = preds[:,1:,:].reshape(-1,42)               # preds_ = [batch_size*text_seq_len, 42]
    text_ = text[1:,:].transpose(0,1).reshape(-1)       # text_ = [batch_size*text_seq_len]
    loss = criterion(preds_, text_.type(torch.long))
    
    # validation forward pass
    with torch.no_grad():
        text_val, vid_val = next(iter(val_data_loader))
        text_val = text_val.to(device)
        vid_val = vid_val.to(device)
        preds_val = model(text=None, vid=vid_val, teacher_force_ratio=0)
        preds_val, text_val = pad_outputs(preds_val,text_val, device=device)   # pad outputs to match dimensions
        preds_val_ = preds_val[:,1:,:].reshape(-1,42)              
        text_val_ = text_val[1:,:].transpose(0,1).reshape(-1)
        loss_val = criterion(preds_val_, text_val_.type(torch.long))

    # keep record of train metrics
    text = itos(text.transpose(0,1))
    preds = itos(terminate_seq_early(preds))
    wer_ = wer(text,preds)
    cer_ = CER(preds, text)
    loss = loss/p['accumulation_steps']
    wer_ = wer_/p['accumulation_steps']
    cer_ = cer_/p['accumulation_steps']
    loss_ += loss.to('cpu')
    wer__ += wer_
    cer__ += cer_
    
    # keep record of validation metrics
    text_val = itos(text_val.transpose(0,1))
    preds_val = itos(terminate_seq_early(preds_val))
    val_wer_ = wer(text_val,preds_val)
    val_cer_ = CER(preds_val, text_val)
    val_loss = loss_val/p['accumulation_steps']
    val_wer_ = val_wer_/p['accumulation_steps']
    val_cer_ = val_cer_/p['accumulation_steps']
    val_loss_ += loss_val.to('cpu')
    val_wer__ += val_wer_
    val_cer__ += val_cer_
    
    # free up GPU space before backpropagation
    del text
    del vid
    del preds_
    del text_
    
    del text_val
    del vid_val
    del preds_val_
    del text_val_
    
    # backwards and optimise
    loss.backward() 
    if (i+1)%p['accumulation_steps'] == 0:
        scheduler.step(loss_)
        optimizer.step()  
        optimizer.zero_grad()
        running_wer += wer__/(p['save_freq']/p['accumulation_steps'])
        running_cer += cer__/(p['save_freq']/p['accumulation_steps'])
        running_loss += loss_/(p['save_freq']/p['accumulation_steps'])
        
        val_running_wer += val_wer__/(p['save_freq']/p['accumulation_steps'])
        val_running_cer += val_cer__/(p['save_freq']/p['accumulation_steps'])
        val_running_loss += val_loss_/(p['save_freq']/p['accumulation_steps'])
        if (i+1)%p['save_freq']==0 or i+1==p['n_steps'] or i==0:
            checkpoint = {'model_state':model.state_dict(), 'optim_state':optimizer.state_dict()}
            torch.save(checkpoint, p['model_save_path'])

            n_steps = p['n_steps']
            print (f'Step [{i+1}/{n_steps}, Train Loss: {running_loss.item()}, Train WER: {running_wer*100}, Train CER: {running_cer*100}')
            print (f'Step [{i+1}/{n_steps}, Val Loss: {val_running_loss.item()}, Val WER: {val_running_wer*100}, Val CER: {val_running_cer*100}')
            # print memory storage
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            print(f'free     : {info.free/10**9}')
            
            training_loss.append(running_loss.item()), training_CER.append(running_cer), training_WER.append(running_wer)
            validation_loss.append(val_running_loss.item()), validation_CER.append(val_running_cer), validation_WER.append(val_running_wer)
            loss_, val_loss_ = 0, 0
            wer__, val_wer__ = 0, 0
            cer__, val_cer__ = 0, 0
            running_wer, val_running_wer = 0, 0
            running_cer, val_running_cer = 0, 0
            running_loss, val_running_loss = 0, 0

            loses = {'training_loss':training_loss,
            'training_CER':training_CER,
            'training_WER':training_WER}
            val_loses = {'val_training_loss':validation_loss,
            'val_training_CER':validation_CER,
            'val_training_WER':validation_WER}
            torch.save(loses, p['train_losses_file'])
            torch.save(val_loses, p['val_losses_file'])

            # live plots
            plt.figure()
            plt.title('Loss curve')
            plt.xlabel('Step')
            plt.ylabel('Loss value')
            plt.plot(np.arange(1,len(training_loss)+1), training_loss)
            plt.plot(np.arange(1,len(validation_loss)+1), validation_loss)
            plt.savefig('Pretrain_Loss_plot')
            plt.close()

            plt.figure()
            plt.title('CER curve')
            plt.xlabel('Step')
            plt.ylabel('CER')
            plt.plot(np.arange(1,len(training_CER)+1), training_CER)
            plt.plot(np.arange(1,len(validation_CER)+1), validation_CER)
            plt.savefig('Pretrain_CER_plot')
            plt.close()

            plt.figure()
            plt.title('WER curve')
            plt.xlabel('Step')
            plt.ylabel('WER')
            plt.plot(np.arange(1,len(training_WER)+1), training_WER)
            plt.plot(np.arange(1,len(validation_WER)+1), validation_WER)
            plt.savefig('Pretrain_WER_plot')
            plt.close()

        else:
            loss_, val_loss_ = 0, 0
            wer__, val_wer__ = 0, 0
            cer__, val_cer__ = 0, 0
