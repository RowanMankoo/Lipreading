"""
Contains useful functions for processing of predictions
"""

from Data import *
from nltk.metrics import edit_distance
from jiwer import wer

def terminate_seq_early(predictions):
    
    """
    Scans predictions for "<EOS>" tokens and if they occur, the rest of sequence 
    is cut off and padded with "<PAD>" tokens 
    """
    # predictions = [batch_size, seq_len, 42]
    EOS_tok = 39
    PAD_tok = 0
    predictions = predictions.argmax(dim=2).to('cpu')
    batch_size, seq_len = predictions.shape
    sequences = []
    
    for i in range(batch_size):
        for indx, character in enumerate(predictions[i]):
            if (character == EOS_tok) and (indx+1 != seq_len):
                predictions[i][indx+1:] = torch.zeros(seq_len-indx-1)
                break
        sequences.append(predictions[i])
    
    return np.stack(sequences, axis=0)


def itos(numericalised_text):
    
    """
    Converts batches of numericalised text back into it's string format
    """
    # numericalised_text = [batch_size, text_seq_len]
    # character level conversion
    vocab = Vocabulary()
    numericalised_text = numericalised_text[:,1:]   # remove "<SOS>" token
    batch_size, _ = numericalised_text.shape
    sentences = []
    for i in range(batch_size):
        sentences.append(np.array([vocab.itos[num.item()] for num in numericalised_text[i]]))
    text = np.stack(sentences, axis=0)
    
    # word level conversion
    sentences = []
    for i in range(batch_size):        
        words = ''
        for character in text[i]:
            if character == '<EOS>':
                break
            words += character
        sentences.append(words)

    return sentences

def CER(predictions, actual):
    """
    Takes two lists of sentences and computes the CER
    """
    total_edits = 0
    total_chars = 0
    for pred_sentence, acc_sentence in zip(predictions,actual):
        
        num_edits = edit_distance(pred_sentence, acc_sentence)
        total_edits += num_edits
        total_chars += len(acc_sentence)
    
    return total_edits/total_chars

    
def pad_outputs(predicted, actual, device):
    """
    Pads either the predicted or actual so that they match in size
    """
    # predicted = [batch_size, pred_seq_len, 42]
    # actual = [acc_seq_len, batch_size]
    
    batch_size = predicted.shape[0]
    pred_seq_len = predicted.shape[1]
    acc_seq_len = actual.shape[0]
    difference = acc_seq_len - pred_seq_len
    
    if difference > 0:
        pad_tensor = torch.zeros(batch_size, difference, 42).to(device)
        pad_tensor[:,:,0] = 1
        padded_preds = torch.cat((predicted,pad_tensor),dim=1)
        
        return padded_preds, actual
        
    else:
        pad_tensor = torch.zeros(-difference,batch_size).to(device)
        padded_acc = torch.cat((actual,pad_tensor),dim=0)
        
        return predicted, padded_acc
