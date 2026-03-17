import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]
def collate_MIL_coords(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	coords = np.vstack([item[2] for item in batch])
	id = [item[3] for item in batch]
	return [img, label,coords,id]


def multi_collate_MIL_coords(batch):
	image = [item[0] for item in batch]
	N_values = torch.tensor([img.shape[0] for img in image])
	max_N = N_values.max().item()
	image = torch.stack([
    		torch.cat([img, torch.zeros(max_N - img.shape[0], img.shape[1])], dim=0)
    		if img.shape[0] < max_N else img
    		for img in image
		])
	label = torch.LongTensor([item[1] for item in batch])
	coords = [item[2] for item in batch]
	coords = torch.stack([
    		torch.cat([torch.tensor(coord), torch.zeros(max_N - coord.shape[0], coord.shape[1])], dim=0)
    		if coord.shape[0] < max_N else torch.tensor(coord)
    		for coord in coords
		])
	if len(batch[0]) == 4:
		slide_id = [item[3] for item in batch]
		return [image, label,coords,N_values,slide_id]
	return [image, label,coords,N_values]
def WSI_data_collate(batch):
	features = torch.stack([feature[0] for feature in batch])
	labels = torch.LongTensor([item[1] for item in batch])
	if len(batch[0]) == 3:
		slide_id = [item[2] for item in batch]
		return [features, labels, slide_id]
	return [features, labels]



def get_caption_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = multi_caption, **kwargs)
	return loader 

def multi_caption(batch):
	caption = [item[0] for item in batch]
	label = torch.LongTensor([item[1] for item in batch])
	name = [item[2] for item in batch]
	return [caption,label,name]

def get_wsi_loader(dataset, batch_size=1, num_workers=1, training = False, weighted = False):
	kwargs = {'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(dataset)
			loader = DataLoader(dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = WSI_data_collate,  **kwargs)
		else:
			loader = DataLoader(dataset, batch_size=batch_size, sampler = RandomSampler(dataset), collate_fn = WSI_data_collate, **kwargs)
	else:
		loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = WSI_data_collate, **kwargs)
	return loader 

def get_coords_id_loader(dataset, batch_size=1, num_workers=1, training = False, weighted = False):
	kwargs = {'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(dataset)
			loader = DataLoader(dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = multi_collate_MIL_coords, **kwargs)
		else:
			loader = DataLoader(dataset, batch_size=batch_size, sampler = RandomSampler(dataset), collate_fn = multi_collate_MIL_coords, **kwargs)
	else:
		loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = multi_collate_MIL_coords, **kwargs)
	return loader 

def get_coords_realid_loader(dataset, batch_size=1, num_workers=1, training = False, weighted = False):
	kwargs = {'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(dataset)
			loader = DataLoader(dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = multi_collate_MIL_id_coords, **kwargs)
		else:
			loader = DataLoader(dataset, batch_size=batch_size, sampler = RandomSampler(dataset), collate_fn = multi_collate_MIL_id_coords, **kwargs)
	else:
		loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = multi_collate_MIL_id_coords, **kwargs)
	return loader 


def multi_caption_image_coords(batch):
	image = [item[0] for item in batch]
	N_values = torch.tensor([img.shape[0] for img in image])
	max_N = N_values.max().item()
	image = torch.stack([
    		torch.cat([img, torch.zeros(max_N - img.shape[0], img.shape[1])], dim=0)
    		if img.shape[0] < max_N else img
    		for img in image
		])
	label = torch.LongTensor([item[2] for item in batch])
	coords = [item[3] for item in batch]
	coords = torch.stack([
    		torch.cat([torch.tensor(coord), torch.zeros(max_N - coord.shape[0], coord.shape[1])], dim=0)
    		if coord.shape[0] < max_N else torch.tensor(coord)
    		for coord in coords
		])
	caption_token = [item[1] for item in batch]
	N_caption_values = torch.tensor([caption.shape[0] for caption in caption_token])
	max_caption= N_caption_values.max().item()
	caption_token = torch.stack([
    		torch.cat([caption, torch.zeros(max_caption - caption.shape[0], caption.shape[1])], dim=0)
    		if caption_token.shape[0] < max_caption else caption_token
    		for caption in caption_token
		])
	return [image, label,coords,N_values,caption_token,N_caption_values]


def get_caption_image_loader(dataset, batch_size=1, num_workers=1, training = False, weighted = False):
	kwargs = {'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(dataset)
			loader = DataLoader(dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = multi_caption_image_coords, **kwargs)
		else:
			loader = DataLoader(dataset, batch_size=batch_size, sampler = RandomSampler(dataset), collate_fn = multi_caption_image_coords, **kwargs)
	else:
		loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = multi_caption_image_coords, **kwargs)
	return loader 

def multi_collate_MIL_id_coords(batch):
	image = [item[0] for item in batch]
	id = [item[3] for item in batch]
	N_values = torch.tensor([img.shape[0] for img in image])
	max_N = N_values.max().item()
	image = torch.stack([
    		torch.cat([img, torch.zeros(max_N - img.shape[0], img.shape[1])], dim=0)
    		if img.shape[0] < max_N else img
    		for img in image
		])
	label = torch.LongTensor([item[1] for item in batch])
	coords = [item[2] for item in batch]
	coords = torch.stack([
    		torch.cat([torch.tensor(coord), torch.zeros(max_N - coord.shape[0], coord.shape[1])], dim=0)
    		if coord.shape[0] < max_N else torch.tensor(coord)
    		for coord in coords
		])
	return [image, label,coords,N_values,id]

def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 16, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 16} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	#print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

