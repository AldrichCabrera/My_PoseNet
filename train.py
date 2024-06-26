#!/usr/bin/python
import os
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from pose_utils import *
from torch.backends import cudnn
from data_loader import get_loader
from torch.optim import lr_scheduler
from model import model_parser, PoseLoss

class Solver():
	def __init__(self, data_loader, image_path, learn_beta):
		self.data_loader = data_loader
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model = model_parser(False, 0.5)
		self.criterion = PoseLoss(self.device, 0.0, -3.0, False)
		# ~ self.print_network(self.model, self.config.model)
		self.data_name = image_path.split('/')[-1]
		self.model_save_path = 'models_%s' % self.data_name

	def print_network(self, model, name):
		num_params = 0
		for param in model.parameters():
			num_params += param.numel()

		print('*' * 20)
		print(name)
		print(model)
		print('*' * 20)

	def train(self):
		self.model = self.model.to(self.device)
		self.criterion = self.criterion.to(self.device)

		if learn_beta:
			optimizer = optim.Adam([{'params': self.model.parameters()},
				{'params': [0.0, -3.0]}],
				lr = 0.0001,
				weight_decay=0.0005)
		else:
			optimizer = optim.Adam(self.model.parameters(),
				lr = 0.0001,
				weight_decay=0.0005)

		scheduler = lr_scheduler.StepLR(optimizer, step_size = 50, gamma=0.1)
		num_epochs = 51

		if not os.path.exists(self.model_save_path):
			os.makedirs(self.model_save_path)

		since = time.time()
		n_iter = 0
		start_epoch = 0

		# Pre-define variables to get the best model
		best_train_loss = 10000
		best_val_loss = 10000
		best_train_model = None
		best_val_model = None

		for epoch in range(start_epoch, num_epochs):
			print('Epoch {}/{}'.format(epoch, num_epochs-1))
			print('-'*20)

			error_train = []
			error_val = []

			for phase in ['train', 'val']:
				if phase == 'train':
					scheduler.step()
					self.model.train()
				else:
					self.model.eval()

				data_loader = self.data_loader[phase]

				for i, (inputs, poses) in enumerate(data_loader):
					inputs = inputs.to(self.device)
					poses = poses.to(self.device)

					optimizer.zero_grad()

					# forward
					pos_out, ori_out = self.model(inputs)

					pos_true = poses[:, :3]
					ori_true = poses[:, 3:]

					ori_out = F.normalize(ori_out, p=2, dim=1)
					ori_true = F.normalize(ori_true, p=2, dim=1)

					loss, _, _ = self.criterion(pos_out, ori_out, pos_true, ori_true)
					loss_print = self.criterion.loss_print[0]
					loss_pos_print = self.criterion.loss_print[1]
					loss_ori_print = self.criterion.loss_print[2]

					if phase == 'train':
						loss.backward()
						optimizer.step()
						n_iter += 1
						error_train.append(loss_print)
					else:
						error_val.append(loss_print)
					print('{}th {} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, phase, loss_print, loss_pos_print, loss_ori_print))

			# For each epoch
			error_train_loss = np.median(error_train) if error_train else float('inf')
			error_val_loss = np.median(error_val) if error_val else float('inf')

			if (epoch+1) % 10 == 0:
				save_filename = self.model_save_path + '/%s_net.pth' % epoch
				# save_path = os.path.join('models', save_filename)
				torch.save(self.model.cpu().state_dict(), save_filename)
				if torch.cuda.is_available():
					self.model.to(self.device)

			if error_train_loss < best_train_loss:
				best_train_loss = error_train_loss
				best_train_model = epoch
				save_filename = self.model_save_path + '/best_train_net.pth'
				torch.save(self.model.cpu().state_dict(), save_filename)
				print(f'Best training model saved at {save_filename}')
				if torch.cuda.is_available():
					self.model.to(self.device)
			if error_val_loss < best_val_loss:
				best_val_loss = error_val_loss
				best_val_model = epoch
				save_filename = self.model_save_path + '/best_net.pth'
				torch.save(self.model.cpu().state_dict(), save_filename)
				if torch.cuda.is_available():
					self.model.to(self.device)

			print('Train and Validaion error {} / {}'.format(error_train_loss, error_val_loss))
			print('=' * 40)
			print('=' * 40)

		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
	cudnn.benchmark = True
	image_path = '/home/aldrich/Downloads/deep_learning/My_PoseNet/KingsCollege'
	metadata_path = '/dataset_train.txt'
	batch_size = 16
	learn_beta = False
	shuffle = True
	data_loader = get_loader(image_path, image_path + metadata_path, 'train', batch_size, shuffle)
	solver = Solver(data_loader, image_path, learn_beta)
	solver.train()
