import numpy as np

import math
import torch
import random
import difflib

from torch import nn
from keras.preprocessing.sequence import pad_sequences
from PIL import Image

def E1(TP, TN, FP, FN):

	accuracy = (TP + TN)/(TP + FP + FN + TN)
	precision = TP/(TP + FP)
	recall = TP/(TP + FN)
	f1_score = 2*(recall*precision)/(recall + precision)
	mcc = (TP * TN - FP * FN)/math.sqrt ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

	return accuracy, f1_score, mcc

def get_pair_cap(idx, length, labels):

	cap_1 = labels[idx]["articles"][0]['caption_modified'].replace('\n','')

	if len(labels[idx]["articles"]) > 1:
		for j in range(len(labels[idx]["articles"])):
			if j != len(labels[idx]["articles"]) - 1:
				cap_2 = labels[idx]["articles"][j + 1]['caption_modified'].replace('\n','')
				if difflib.SequenceMatcher(None, cap_1, cap_2).ratio() < 0.5:
					return cap_1, cap_2, 1

			else:
				r = random.choice([k for k in range(0,length) if k not in [idx]])
				cap_2 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified'].replace('\n','')
				return cap_1, cap_2, -1


	else:
		r = random.choice([k for k in range(0,length) if k not in [idx]])
		cap_2 = labels[r]["articles"][random.randint(0, len(labels[r]["articles"]) - 1)]['caption_modified'].replace('\n','')
		return cap_1, cap_2, -1

def validation_loss(model, labels, cosine_loss, tokenizer, transform):

	# length = len(labels)
	random.shuffle(labels)
	length = 1000
	iters = int(length/1)
	loss = 0

	model.eval()

	for k in range(iters):
		target = []

		for i in range(1):
			idx = k
			path = '../Data/' +  labels[idx]['img_local_path']

			img = Image.open(path)
			img = transform(img)
			img = torch.unsqueeze(img, 0)

			cap_1, cap_2, y = get_pair_cap(idx, length, labels)
			target.append(y)

		tokens_1, mask_1 = word_tokenize(cap_1, tokenizer)
		tokens_2, mask_2 = word_tokenize(cap_2, tokenizer)
		targets = torch.Tensor(target).to(torch.device("cuda:0"))

		with torch.no_grad():
			x_1 = model(tokens_1.to(torch.device("cuda:0")), img.to(torch.device("cuda:0")), mask_1.to(torch.device("cuda:0")))
			x_2 = model(tokens_2.to(torch.device("cuda:0")), img.to(torch.device("cuda:0")), mask_2.to(torch.device("cuda:0")))
		loss_ = cosine_loss(x_1, x_2, targets)
		loss += loss_.item()


	model.train()
	return loss/iters

def word_tokenize(tokens, tokenizer):

	tokens = tokenizer.encode_plus(tokens, add_special_tokens=True,
	max_length=512, truncation=True,
	padding="max_length", return_tensors='pt')

	input_ids = tokens['input_ids'].type(torch.LongTensor)
	# token = torch.Tensor(tokens).type(torch.LongTensor)

	train_masks = tokens['attention_mask'].type(torch.LongTensor)
	return input_ids, train_masks
