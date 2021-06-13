import sys
import json
import numpy as np
import random as rn
import torch

from torch import nn
from torchvision import transforms
from transformers import BertTokenizer
from efficientnet_pytorch import EfficientNet
from sentence_transformers import SentenceTransformer, util
from scipy import spatial
from PIL import Image

from torch import optim
from model import *
from simple_tools import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
labels = []

for line in f:
	labels.append(json.loads(line))

length = len(labels)

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

model = EfficientNet.from_pretrained('efficientnet-b3')
model = BertBinaryClassifier(model).to(torch.device("cuda:0"))

# model_bert = SentenceTransformer('stsb-mpnet-base-v2').to(torch.device("cuda:0"))

model.load_state_dict(torch.load("Model/model_0.pth"))

TP = TN = FP = FN = idx = total_cos_neg = total_cos_pos = pos = neg =0

label_context = []

thresholds = np.arange(0, 1, 0.01)
# cos_thresholds = np.arange(0, 1, 0.05)

for label in labels:

	# emb_sent_1 = model_bert.encode(label['caption1_modified'])
	# emb_sent_2 = model_bert.encode(label['caption2_modified'])
	# cos_sim = spatial.distance.cosine(emb_sent_1, emb_sent_2)

	path = '../Data/' +  label['img_local_path']
	path = path.replace(':', '_')
	img = Image.open(path)
	img = transform(img)
	img = torch.unsqueeze(img, 0)

	tokens_1, mask_1 = word_tokenize(label['caption1_modified'], tokenizer)
	tokens_2, mask_2 = word_tokenize(label['caption2_modified'], tokenizer)

	with torch.no_grad():

		x_1 = model(tokens_1.to(torch.device("cuda:0")), img.to(torch.device("cuda:0")), mask_1.to(torch.device("cuda:0")))
		x_2 = model(tokens_2.to(torch.device("cuda:0")), img.to(torch.device("cuda:0")), mask_2.to(torch.device("cuda:0")))

	cos = spatial.distance.cosine(x_1.cpu().numpy(), x_2.cpu().numpy())
	label_context.append([label['context_label'], cos])


accs = []

for threshold in thresholds:
	TP = TN = FP = FN = 0

	for label, cos in label_context:

		if cos < threshold and label == 0:
			TN += 1
		elif cos > threshold and label == 0:
			FP += 1
		elif cos > threshold and label == 1:
			TP += 1
		elif cos < threshold and label == 1:
			FN += 1

	if TP != 0 and TN != 0 and FP != 0 and FN != 0:
		acc, f1, mcc = E1(TP, TN, FP, FN)
		accs.append(acc)
		print("Threshold: {} \t Accuracy: {}".format(threshold, acc))

print("Accuracy: ",accs[np.argmax(np.asarray(accs))])