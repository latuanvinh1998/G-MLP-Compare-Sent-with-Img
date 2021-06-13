import torch
import torch.nn.functional as F

from torch import nn
from pytorch_pretrained_bert import BertModel


class gMLP_Classification(nn.Module):
	def __init__(self, net):
		super().__init__()
		self.net = net
		self._fc = nn.Linear(1024, 256)
		self.fc_1 = nn.Linear(256, 2)

	def forward(self, x1, x2):
		x_1 = self.net(x1)
		x_2 = self.net(x2)
		emb = torch.cat((torch.mean(x_1, 1), torch.mean(x_2, 1)), 1)

		emb = self._fc(emb)
		out = self.fc_1(emb)
		return out


class BertBinaryClassifier(nn.Module):
    def __init__(self, net):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.net = net

        # self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(768, 512)
        self.linear_2 = nn.Linear(1536, 512)
        self.linear = nn.Linear(512, 256)
    
    def forward(self, tokens, img, masks=None):
        _, emb_cap = self.bert(tokens, attention_mask=masks)
        emb_img = self.net(img)
        
        # dropout_output = self.dropout(pooled_output)
        emb_cap = self.linear_1(emb_cap)
        emb_img = self.linear_2(emb_img)
        # emb = torch.cat((emb_cap, emb_img), 1)
        out = self.linear(emb_cap + emb_img)

        return out