"""
Code used to generate the plots, note that if you wish to recreate plots you need to manually load in your loss/WER/CER data
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


# first plot 
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey='none')
fig.suptitle('Curriculum\'s 1 to 4 loss curve\'s', fontsize=35)
fig.text(0.5, 0.03, 'Iterations', ha='center', fontsize=20)
fig.set_size_inches(18.5, 10.5)

ax1.plot(training_loss1)
ax1.set_title('n_words = 1', y=-0.1)
ax1.set_xticks([])
ax1.set_ylabel('Loss',fontsize=20)

ax2.plot(training_loss2)
ax2.set_title('n_words = 1',y=-0.1)
ax2.set_xticks([])

ax3.plot(training_loss3)
ax3.set_title('n_words = 2',y=-0.1)
ax3.set_xticks([])
ax3.set_ylabel('Loss', fontsize=20)

ax4.plot(training_loss4)
ax4.set_title('n_words = 3',y=-0.1)
ax4.set_xticks([])
fig.savefig('loss curriculum1-4')
plt.close(fig)




# second plot
b = training_loss6+training_loss7+training_loss8 +training_loss9 +training_loss10
iters = [len(training_loss6), len(training_loss7), len(training_loss8), len(training_loss9), len(training_loss10)]
vlines = np.cumsum(iters)
plt.plot(b)
for val in vlines:
    plt.axvline(x=val, color='red', linestyle='dashed')
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Loss', fontsize=20)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 6)
fig.suptitle('Curriculum\'s 6 to 10 loss curve', fontsize=32)
fig.savefig('loss curriculum1-6')
plt.close(fig)




# third plot
fig, (ax1,ax2) = plt.subplots(2,1, sharey='none')
fig.suptitle('Curriculum 15 CER/WER curves', fontsize=35)
ax1.plot(training_CER2+training_CER3)
ax1.plot(validation_CER2+validation_CER3)
ax1.set_ylabel('CER', fontsize=20)

ax2.plot(training_WER2+training_WER3)
ax2.plot(validation_WER2+validation_WER3)
ax2.set_ylabel('WER', fontsize=20)
ax2.set_xlabel('Iterations', fontsize=20)
fig.set_size_inches(18.5, 10.5)
fig.savefig('curriculum 15')
plt.close(fig)