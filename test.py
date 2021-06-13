import sys
import json
import numpy as np
import random as rn
import torch

from torch import nn
from torchvision import transforms
from transformers import BertTokenizer
from efficientnet_pytorch import EfficientNet
from PIL import Image

from torch import optim
from model import *
from simple_tools import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model = EfficientNet.from_pretrained('efficientnet-b3')
model = BertBinaryClassifier(model).to(torch.device("cuda:0"))

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=4e-6)
cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.3)

# path = '../Data/' +  validation[0]['img_local_path']
# img = Image.open(path)
# img = transform(img)
# img = torch.unsqueeze(img, 0)
# print(model(img).shape)
# raise Exception

# cap_1, cap_2, y = get_pair_cap(0, 100, validation)
# ids, mask = word_tokenize(cap_1, tokenizer)

# print(model(ids.to(torch.device("cuda:0")), img.to(torch.device("cuda:0")), mask.to(torch.device("cuda:0"))).shape)

# print(validation_loss(model, validation, cosine_loss, tokenizer, transform))
model.train()
while EPOCH < 20:
	random.shuffle(labels)

	for k in range(iters):
		target = []

		for i in range(BATCH_SIZE):
			idx = BATCH_SIZE*k + i

			path = '../Data/' +  labels[idx]['img_local_path']

			img = Image.open(path)
			img = transform(img)
			img = torch.unsqueeze(img, 0)

			cap_1, cap_2, y = get_pair_cap(idx, length, labels)
			target.append(y)

		tokens_1, mask_1 = word_tokenize(cap_1, tokenizer)
		tokens_2, mask_2 = word_tokenize(cap_2, tokenizer)
		targets = torch.Tensor(target).to(torch.device("cuda:0"))

		optimizer.zero_grad()
		x_1 = model(tokens_1.to(torch.device("cuda:0")), img.to(torch.device("cuda:0")), mask_1.to(torch.device("cuda:0")))
		x_2 = model(tokens_2.to(torch.device("cuda:0")), img.to(torch.device("cuda:0")), mask_2.to(torch.device("cuda:0")))

		loss = cosine_loss(x_1, x_2, targets)
		Accumulate_Loss += loss.item()
		loss.backward()
		optimizer.step()

		GLOBAL_STEP += 1

		if GLOBAL_STEP % 100 == 0:
			a_loss = Accumulate_Loss/100
			print("Epoch: %d === Global Step: %d === Loss: %.3f" %(EPOCH, GLOBAL_STEP, a_loss))
			Accumulate_Loss = 0

		if GLOBAL_STEP % 1000 == 0:
			print("Validation.......")
			val = validation_loss(model, validation, cosine_loss, tokenizer, transform)
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