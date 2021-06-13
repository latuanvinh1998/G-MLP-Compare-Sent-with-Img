from g_mlp_pytorch import gMLP
from g_mlp_pytorch.autoregressive_wrapper import AutoregressiveWrapper

import random
import os
import json
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from einops.layers.torch import Rearrange, Reduce
from network import *
from simple_tools import *

# constants

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

os.makedirs("Model_MLP/", exist_ok=True)

f_train = open('../Data/mmsys_anns/train_data.json')
labels = []

f_val = open('../Data/mmsys_anns/val_data.json')
validation = []

for line in f_train:
    labels.append(json.loads(line))

for line in f_val:
    validation.append(json.loads(line))

length = len(labels)
iters = int(length/8)

model_mlp = gMLP(
    num_tokens = 256,
    dim = 512,
    seq_len = 768,
    depth = 8,
    causal = True
)


model_mlp.to_logits = nn.Identity()
model = gMLP_Classification(model_mlp).to(torch.device("cuda:0"))

model.load_state_dict(torch.load('Model_MLP/model_5.pth'))

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=4e-6)
criterion = nn.CrossEntropyLoss()

epoch = global_step = accumulate_loss = 0
pre_val = 5

model.train()

while epoch < 1000:
    random.shuffle(labels)
    for k in range(iters):
        for i in range(8):
            idx = 8*k + i

            cap_1, cap_2, y = get_pair_cap(idx, length, labels, i)
            label = torch.LongTensor([y]).to(torch.device("cuda:0"))

            cap_1 = bytearray(cap_1.encode())
            cap_2 = bytearray(cap_2.encode())

            cap_1 = np.frombuffer(cap_1, dtype=np.uint8)
            cap_2 = np.frombuffer(cap_2, dtype=np.uint8)

            if cap_1.shape[0] > 768:
                cap_1 = cap_1[0:700]

            if cap_2.shape[0] > 768:
                cap_2 = cap_2[0:700]

            cap_1 = torch.unsqueeze(torch.from_numpy(cap_1).long(), 0).to(torch.device("cuda:0"))
            cap_2 = torch.unsqueeze(torch.from_numpy(cap_2).long(), 0).to(torch.device("cuda:0"))

            score = model(cap_1, cap_2)

            optimizer.zero_grad()

            loss = criterion(score, label)

            loss.backward()
            optimizer.step()

            accumulate_loss += loss.item()

            global_step += 1

            if global_step % 800 == 0:
                print("Epoch: %d === Global Step: %d === Loss: %.3f" %(epoch, global_step, accumulate_loss/800))
                accumulate_loss = 0

            if global_step % 8000 == 0:
                print("Validation.......")
                val = validation_loss(model, validation, criterion)
                print("Epoch: %d === Global Step: %d === Validation Loss: %.3f" %(epoch, global_step, val))

                if val < pre_val:

                    torch.save(model.state_dict(), 'Model_MLP/model_{}.pth'.format(epoch))
                    torch.save(optimizer.state_dict(), 'Model_MLP/optimizer_{}.pth'.format(epoch))

                    txt = open('Model_MLP/stat_{}.txt'.format(epoch), 'w')
                    txt.write('Loss: %.3f \n'%(loss))
                    txt.write('Validation: %.3f'%(val))
                    txt.close()

                    pre_val = val

    epoch += 1

