import sys
import json
import numpy as np
import random as rn
import torch

from pytorch_pretrained_bert import BertModel
from torch import nn
from torchnlp.datasets import imdb_dataset
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import optim
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
from model import *
from simple_tools import *


BATCH_SIZE = 1
pre_val = 1
EPOCH = TRAIN_LOSS = GLOBAL_STEP = Accumulate_Loss = 0

f_train = open('../Data/mmsys_anns/train_data.json')
labels = []

f_val = open('../Data/mmsys_anns/val_data.json')
validation = []

for line in f_train:
    labels.append(json.loads(line))

for line in f_val:
    validation.append(json.loads(line))

length = len(labels)
iters = int(length/BATCH_SIZE)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertBinaryClassifier().cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()


model.train()
while EPOCH < 20:
    random.shuffle(labels)

    for k in range(iters):
        caps_1 = []
        caps_2 = []
        target = []

        for i in range(BATCH_SIZE):
            idx = BATCH_SIZE*k + i
            cap_1, cap_2, y = get_pair_cap(idx, length, labels)
            caps_1.append(cap_1)
            caps_2.append(cap_2)
            target.append(y)

        tokens_1, mask_1 = word_tokenize(caps_1, tokenizer)
        tokens_2, mask_2 = word_tokenize(caps_2, tokenizer)
        targets = torch.Tensor(target).type(torch.LongTensor).to(device)

        theta = model(tokens_1.to(device), tokens_2.to(device), mask_1.to(device), mask_2.to(device))
        loss = criterion(theta, targets)

        Accumulate_Loss += loss
        model.zero_grad()
        loss.backward()

        # clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()

        GLOBAL_STEP += 1

        if GLOBAL_STEP % 100 == 0:
            a_loss = Accumulate_Loss/100
            print("Epoch: %d === Global Step: %d === Loss: %.3f" %(EPOCH, GLOBAL_STEP, a_loss))
            Accumulate_Loss = 0

        if GLOBAL_STEP % 1000 == 0:
            print("Validation.......")
            val = validation_loss(model, validation, criterion, tokenizer)
            print("Epoch: %d === Global Step: %d === Validation Loss: %.3f" %(EPOCH, GLOBAL_STEP, val))

            if val < pre_val:
                torch.save(model.state_dict(), 'Model/model_{}.pth'.format(EPOCH))
                torch.save(optimizer.state_dict(), 'Model/optimizer_{}.pth'.format(EPOCH))

                txt = open('Model/stat_{}.txt'.format(EPOCH), 'w')
                txt.write('Loss: %.3f \n'%(a_loss))
                txt.write('Validation: %.3f'%(val))
                txt.close()

                pre_val = val


    EPOCH += 1
