from Data import *
from final_model import *
from utils import *
import matplotlib.pyplot as plt

p = {'batch_size':1,  
     'pad_indx':0,
     'file_directory':"/data/fast/rxm064/temp/mvlrs_v1/main",
     'index_directory':"/data/fast/rxm064/temp/mvlrs_v1/train.txt",
     'model_save_path':'/data/fast/rxm064/temp/model.pth'}

# load data
dataset = LRS2Dataset(file_directory=p['file_directory'],
                            index_directory=p['index_directory'])
data_loader = DataLoader(dataset=dataset, 
                         batch_size=p['batch_size'],
                         shuffle=True, 
                         collate_fn=MyCollate())

# load model
device = torch.device('cpu')
model = Lipreading(device=device)
model.to(device)
checkpoint = torch.load(p['model_save_path'])
model.load_state_dict(checkpoint['model_state'])

# generate 1 example
text, vid = next(iter(data_loader))
text = text.to(device)
vid = vid.to(device)
vid.requires_grad_()
model.eval()

# generate prediction
preds = model(text,vid, teacher_force_ratio=0)

# loop through individual character predictions 
seq_len = preds.shape[1]
temp = []
for char in range(0, seq_len):
    indx = preds[0,char,:].argmax()
    score = preds[0,1,indx]
    score.backward(retain_graph=True)

    # Genereate saliency map
    saliency_char = vid.grad.abs().squeeze()   # saliency = [original_vid_seq_len,112,112]
    temp.append(saliency_char.numpy())
# produce global saliency map
saliency = np.stack(temp)
saliency = np.mean(np.squeeze(saliency),axis=0)
model.train()

vid = vid.detach().numpy()
text = itos(text.transpose(0,1))



# plot saliency maps, for arbitray number of frames
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
#fig.suptitle('Saliency Maps')

ax1.imshow(vid[0,0,:,:], cmap='gray')
ax1.imshow(saliency[0,:,:], cmap=plt.cm.hot,alpha=0.5)
ax1.axis('off')

ax2.imshow(vid[0,5,:,:], cmap='gray')
ax2.imshow(saliency[5,:,:], cmap=plt.cm.hot,alpha=0.5)
ax2.axis('off')

ax3.imshow(vid[0,10,:,:], cmap='gray')
ax3.imshow(saliency[10,:,:], cmap=plt.cm.hot,alpha=0.5)
ax3.axis('off')

ax4.imshow(vid[0,15,:,:], cmap='gray')
ax4.imshow(saliency[15,:,:], cmap=plt.cm.hot,alpha=0.5)
ax4.axis('off')

ax5.imshow(vid[0,20,:,:], cmap='gray')
ax5.imshow(saliency[20,:,:], cmap=plt.cm.hot,alpha=0.5)
ax5.axis('off')

fig.savefig('Saliency maps')

text = itos(text.transpose(0,1))