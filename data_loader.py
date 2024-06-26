#!/usr/bin/python
import os
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
	def __init__(self, image_path, metadata_path, mode, transform, num_val=100):
		self.image_path = image_path
		self.metadata_path = metadata_path
		self.mode = mode
		self.transform = transform
		raw_lines = open(self.metadata_path, 'r').readlines()
		self.lines = raw_lines[3:]
		print("=======================================================")
		print("Len lines: ", self.lines.__len__())

		self.test_filenames = []
		self.test_poses = []
		self.train_filenames = []
		self.train_poses = []

		for i, line in enumerate(self.lines):
			splits = line.split()
			filename = splits[0]
			values = splits[1:]
			values = list(map(lambda x: float(x.replace(",", "")), values))

			filename = os.path.join(self.image_path, filename)
			# ~ print("filename: ", filename)
			# ~ print("values: ", values)

			if self.mode == 'train':
				self.train_filenames.append(filename)
				self.train_poses.append(values)
			elif self.mode == 'test':
				self.test_filenames.append(filename)
				self.test_poses.append(values)
			elif self.mode == 'val':
				self.test_filenames.append(filename)
				self.test_poses.append(values)
				if i > num_val:
					break

		if self.mode == 'train':
			self.num_train = self.train_filenames.__len__()
			print("Number of Train", self.num_train)

		elif self.mode in ['val', 'test']:
			self.num_test = self.test_filenames.__len__()
			self.num_val = self.test_filenames.__len__()
			print("Number of Test", self.num_test)
			print("Number of Val", self.num_test)

	def __getitem__(self, index):
		if self.mode == 'train':
			image = Image.open(self.train_filenames[index])
			pose = self.train_poses[index]
		elif self.mode in ['val', 'test']:
			image = Image.open(self.test_filenames[index])
			pose = self.test_poses[index]
		return self.transform(image), torch.Tensor(pose)

	def __len__(self):
		if self.mode == 'train':
			num_data = self.num_train
		elif self.mode in ['val', 'test']:
			num_data = self.num_test
		return num_data

def get_loader(image_path, metadata_path, mode, batch_size, is_shuffle=False, num_val=100):
	img_size = 300
	img_crop = 299

	if mode == 'train':
		transform = transforms.Compose([
			transforms.Resize(img_size),
			transforms.RandomCrop(img_crop),
			transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),            
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

		datasets = {'train': CustomDataset(image_path, metadata_path, 'train', transform, num_val),
					'val': CustomDataset(image_path, metadata_path, 'val', transform, num_val)}
		data_loaders = {'train': DataLoader(datasets['train'], batch_size, is_shuffle, num_workers=4),
						'val': DataLoader(datasets['val'], batch_size, is_shuffle, num_workers=4)}

	elif mode == 'test':
		transform = transforms.Compose([
			transforms.Resize(img_size),
			transforms.CenterCrop(img_crop),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

		batch_size = 1
		is_shuffle = False
		dataset = CustomDataset(image_path, metadata_path, 'test', transform)
		data_loaders = DataLoader(dataset, batch_size, is_shuffle, num_workers=4)
	return data_loaders
