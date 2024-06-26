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
		self.data_name = image_path.split('/')[-1]
		self.model_save_path = 'models_%s' % self.data_name

	def test(self):
		self.model = self.model.to(self.device)
		self.model.eval()

		test_model_path = self.model_save_path + '/9_net.pth'
		print('Load pretrained model: ', test_model_path)
		self.model.load_state_dict(torch.load(test_model_path))

		total_pos_loss = 0
		total_ori_loss = 0
		pos_loss_arr = []
		ori_loss_arr = []
		true_pose_list = []
		estim_pose_list = []
		num_data = len(self.data_loader)

		for i, (inputs, poses) in enumerate(self.data_loader):
			print(i)
			inputs = inputs.to(self.device)

			# forward
			pos_out, ori_out = self.model(inputs)
			pos_out = pos_out.squeeze(0).detach().cpu().numpy()
			ori_out = F.normalize(ori_out, p=2, dim=1)
			ori_out = quat_to_euler(ori_out.squeeze(0).detach().cpu().numpy())
			print('pos out', pos_out)
			print('ori_out', ori_out)

			pos_true = poses[:, :3].squeeze(0).numpy()
			ori_true = poses[:, 3:].squeeze(0).numpy()
			ori_true = quat_to_euler(ori_true)
			print('pos true', pos_true)
			print('ori true', ori_true)

			loss_pos_print = array_dist(pos_out, pos_true)
			loss_ori_print = array_dist(ori_out, ori_true)
			true_pose_list.append(np.hstack((pos_true, ori_true)))

			if loss_pos_print < 20:
				estim_pose_list.append(np.hstack((pos_out, ori_out)))

			print(pos_out)
			print(pos_true)

			total_pos_loss += loss_pos_print
			total_ori_loss += loss_ori_print

			pos_loss_arr.append(loss_pos_print)
			ori_loss_arr.append(loss_ori_print)
			print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))

		position_error = np.median(pos_loss_arr)
		rotation_error = np.median(ori_loss_arr)
		print('=' * 20)
		print('Overall median pose errer {:.3f} / {:.3f}'.format(position_error, rotation_error))
		print('Overall average pose errer {:.3f} / {:.3f}'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))

if __name__ == '__main__':
	cudnn.benchmark = True
	image_path = '/home/aldrich/Downloads/deep_learning/My_PoseNet/KingsCollege'
	metadata_path = '/dataset_test.txt'
	batch_size = 16
	learn_beta = False
	shuffle = True
	data_loader = get_loader(image_path, image_path + metadata_path, 'test', batch_size, shuffle)
	solver = Solver(data_loader, image_path, learn_beta)
	solver.test()