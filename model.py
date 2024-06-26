#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def model_parser(fixed_weight=False, dropout_rate=0.0):
	base_model = models.inception_v3(pretrained=True)
	network = GoogleNet(base_model, fixed_weight, dropout_rate)
	return network

# Loss Criterion
class PoseLoss(nn.Module):
	def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
		super(PoseLoss, self).__init__()
		self.learn_beta = learn_beta

		if not self.learn_beta:
			self.sx = 0
			self.sq = -6.25

		self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
		self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)
		self.loss_print = None

	def forward(self, pred_x, pred_q, target_x, target_q):
		pred_q = F.normalize(pred_q, p=2, dim=1)
		loss_x = F.l1_loss(pred_x, target_x)
		loss_q = F.l1_loss(pred_q, target_q)

		loss = torch.exp(-self.sx)*loss_x + self.sx + torch.exp(-self.sq)*loss_q + self.sq
		self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

		return loss, loss_x.item(), loss_q.item()

class GoogleNet(nn.Module):
	def __init__(self, base_model, fixed_weight=False, dropout_rate = 0.0):
		super(GoogleNet, self).__init__()
		self.dropout_rate = dropout_rate

		model = []
		model.append(base_model.Conv2d_1a_3x3)
		model.append(base_model.Conv2d_2a_3x3)
		model.append(base_model.Conv2d_2b_3x3)
		model.append(nn.MaxPool2d(kernel_size=3, stride=2))
		model.append(base_model.Conv2d_3b_1x1)
		model.append(base_model.Conv2d_4a_3x3)
		model.append(nn.MaxPool2d(kernel_size=3, stride=2))
		model.append(base_model.Mixed_5b)
		model.append(base_model.Mixed_5c)
		model.append(base_model.Mixed_5d)
		model.append(base_model.Mixed_6a)
		model.append(base_model.Mixed_6b)
		model.append(base_model.Mixed_6c)
		model.append(base_model.Mixed_6d)
		model.append(base_model.Mixed_6e)
		model.append(base_model.Mixed_7a)
		model.append(base_model.Mixed_7b)
		model.append(base_model.Mixed_7c)
		self.base_model = nn.Sequential(*model)

		if fixed_weight:
			for param in self.base_model.parameters():
				param.requires_grad = False

		# Out 2
		self.pos2 = nn.Linear(2048, 3, bias=True)
		self.ori2 = nn.Linear(2048, 4, bias=True)

	def forward(self, x):
		x = self.base_model(x) # 299 x 299 x 3
		x = F.avg_pool2d(x, kernel_size=8) # 8 x 8 x 2048
		x = F.dropout(x, p=self.dropout_rate, training=self.training) # 1 x 1 x 2048
		x = x.view(x.size(0), -1) # 1 x 1 x 2048
		pos = self.pos2(x) # 3
		ori = self.ori2(x) # 4
		return pos, ori
