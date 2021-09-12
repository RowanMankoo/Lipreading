"""
Custom Dataset class to load text and videos into required tensor formats
"""
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:
    def __init__(self):
        self.stoi = {"<PAD>":0, " ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34,
                     "4":38, "7":36, "6":35, "9":31, "8":33, "A":5, "C":17,
                     "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9,
                     "K":24, "J":25, "M":18, "L":11, "O":4, "N":7, "Q":27,
                     "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23,
                     "Y":14, "X":26, "Z":28, "<EOS>":39, "<UNK>":40, "<SOS>":41}
        self.itos = {0:"<PAD>", 1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5",
                     38:"4", 36:"7", 35:"6", 31:"9", 33:"8", 5:"A", 17:"C",
                     20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H",
                     24:"K", 25:"J", 18:"M", 11:"L", 4:"O", 7:"N", 27:"Q",
                     21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V",
                     14:"Y",26:"X", 28:"Z", 39:"<EOS>", 40:"<UNK>", 41:"<SOS>"}
        
    def numericalise(self, text):
        # input text in string form
        tokenised_text = list(text)
        numericalised_text = [self.stoi[tok] if tok in self.stoi else self.stoi["<UNK>"]
                             for tok in tokenised_text]
        final_text = [self.stoi["<SOS>"]]
        final_text += numericalised_text
        final_text.append(self.stoi["<EOS>"])
        
        return final_text
        
        
def load_vid(VideoFile):
    """
    Convert video into black and white, and store in numpy array
    """
    cap = cv2.VideoCapture(VideoFile)
    frames = []
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    video = np.stack(frames, axis=0)  # video = [frames, width, height] 
    
    return video
    

def prepare_train_data(TextFile, VideoFile):
    
    # load in text file
    vocab = Vocabulary()
    with open(TextFile) as file:
        phrase = file.readlines()
    phrase = phrase[0][7:].replace('\n','')
    numericalised_phrase = vocab.numericalise(phrase)    
    numericalised_phrase = torch.FloatTensor(numericalised_phrase)
    
    # load data video
    vid = load_vid(VideoFile)    
    
    # numericalised_phrase = [text_seq_len]
    # vid = [frames, width, height]
    
    return numericalised_phrase, vid
    
    
def prepare_pretrain_data(TextFile, VideoFile, num_words, video_FPS=25):
    
    # load in text file
    with open(TextFile) as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    phrase = lines[0][7:]
    words = phrase.split(" ")
    
    if len(words) <= num_words:
        phrase_subsequence = phrase
        vid = load_vid(VideoFile)

    else:
        # create list of possible subsequences of num_words length long and randomly select one to train on
        word_subsequences = [' '.join(words[i:i+num_words]) for i in range(len(words)-num_words+1)]
        indx = np.random.randint(len(word_subsequences))  
        phrase_subsequence = word_subsequences[indx]

        # determine video start/end times
        video_start_time = float(lines[4+indx].split(' ')[1])
        video_end_time = float(lines[4+indx+num_words-1].split(' ')[2])
   
        # load video and crop it accordingly
        vid = load_vid(VideoFile)
        start = int(np.floor(video_FPS*video_start_time))
        end = int(np.ceil(video_FPS*video_end_time))
        vid = vid[start:end]
        

    # numericalise text
    vocab = Vocabulary()
    numericalised_phrase = vocab.numericalise(phrase_subsequence)
    numericalised_phrase = torch.FloatTensor(numericalised_phrase)
    
    # numericalised_phrase = [text_seq_len]
    # vid = [frames, width, height]
    
    return numericalised_phrase, vid 
    

def vid_preprocessing(vid, y_crop=(50,110), x_crop=(40,120)):
    
    """
    Crop a 60x80 region centered around the mouth, and then rescale to 112x112 and normalise values 
    """
    y1, y2 = y_crop
    x1, x2 = x_crop
    frames= []
    
    # crop relevant ROI and resize to (122,122)
    for i in range(vid.shape[0]):
        frame = vid[i] 
        frame = frame[y1:y2, x1:x2]
        frame = cv2.resize(frame, (112,112))
        frames.append(frame)
    cropped_vid = np.stack(frames, axis=0) 
    
    # normalise values between 0,1
    cropped_vid = cropped_vid/255
    cropped_vid = torch.FloatTensor(cropped_vid)
    
    return cropped_vid
    
    
class LRS2Pretrain(Dataset):
    
    """
    Custom dataset class for LRS2 pretrain dataset
    
    
    Args:
        file_directory (str): LRS2 dataset directory
        num_words (int): Number of words to be used in each subsequence for current curriculum
        index_directory (str): Directory path of index file names 
    
    """
    def __init__(self, file_directory, num_words, index_directory):
        super(LRS2Pretrain, self).__init__()
        self.num_words = num_words
        
        with open(index_directory) as f:          
            lines = f.readlines()
        lines = [word.replace('\n','') for word in lines]
        
        self.video_files = [file_directory+'/'+line+'.mp4' for line in lines]
        self.text_files = [file_directory+'/'+line+'.txt' for line in lines]
        assert len(self.video_files) == len(self.text_files) 
                
    def __getitem__(self,index):
        text, vid = prepare_pretrain_data(self.text_files[index],self.video_files[index],
                                          num_words=self.num_words)
        vid = vid_preprocessing(vid)
        
        # text = [text_seq_len]
        # vid = [vid_seq_len, 112, 112]
        
        return text, vid
    
    def __len__(self):
        return len(self.video_files)
        
        
class LRS2Dataset(Dataset):
    
    """
    Custom dataset class for LRS2 train dataset
    
    Args:
        file_directory (str): LRS2 dataset directory
        index_directory (str): directory path of index file names 
    
    """
    def __init__(self, file_directory, index_directory):
        super(LRS2Dataset, self).__init__()     
        # read in labels of current dataset
        with open(index_directory) as f:          
            lines = f.readlines()
        lines = [word.replace('\n','') for word in lines]
        
        self.video_files = [file_directory+'/'+line+'.mp4' for line in lines]
        self.text_files = [file_directory+'/'+line+'.txt' for line in lines]
        assert len(self.video_files) == len(self.text_files) 
    
    def __getitem__(self, index):       
        txt, vid = prepare_train_data(self.text_files[index],self.video_files[index])
        vid = vid_preprocessing(vid)
        
        # text = [text_seq_len]
        # vid = [vid_seq_len, 112, 112]
        
        return txt, vid
        
    def __len__(self):
        return len(self.video_files)
        
        
class MyCollate:
    def __call__(self, batch):
        # text batching (padding)
        text_batch = [item[0] for item in batch]
        text_batch = pad_sequence(text_batch, batch_first=False, padding_value=0)
        # video batching
        vid_batch = pad_sequence([item[1] for item in batch], batch_first=True)
        
        return text_batch, vid_batch

