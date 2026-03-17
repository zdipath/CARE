import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from datasets import load_dataset, concatenate_datasets, load_from_disk

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):

	splits = []
	for i in range(len(split_datasets)):
		if split_datasets[i] != None:
			splits.append(split_datasets[i].slide_data['slide_id'])
	#splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) if dset is not None else 0 for dset in split_datasets ], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		suffix = '0_1024',
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		data_format = '.svs'
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		self.suffix = suffix
		self.data_format = data_format
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		patients = sorted(patients, key=natural_sort_key)
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				#label = stats.mode(label)[0]
				label, counts = np.unique(label, return_counts=True)
				label = label[np.argmax(counts)]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)

		mask_not_in_label_dict = ~data['label'].isin(label_dict.keys())

		data = data[~mask_not_in_label_dict]

		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 
			#print("len(case_id list):", len(self.patient_data['case_id']))
			for split in range(len(ids)): 
				#print(split)
				for idx in ids[split]:
					#if idx>300:
					#	print(idx)
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			slide_ids = self.slide_data['slide_id'].to_numpy()


			split_ids = np.array(split.tolist())

			common_dtype = np.result_type(slide_ids.dtype, split_ids.dtype)

			slide_ids = slide_ids.astype(common_dtype)
			split_ids = split_ids.astype(common_dtype)

			mask = np.isin(slide_ids, split_ids)
			#mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(self, df_slice, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format,suffix = self.suffix)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(self, df_slice, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format,suffix = self.suffix)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(self, train_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format,suffix = self.suffix)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(self, val_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format,suffix = self.suffix)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(self, test_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format,suffix = self.suffix)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			if 'val' in all_splits.columns:
				val_split = self.get_split_from_df(all_splits, 'val')
			else:
				val_split = None
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		data_format = '.svs',
		**kwargs):
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False
		self.return_slide_id = False
		self.data_format = data_format
	def load_from_h5(self, toggle):
		self.use_h5 = toggle
	def return_slideid(self):
		self.return_slide_id = True
	def return_feature(self,slide_id,i = None):
		new_list = ["/".join(s.split("/")[-1:]).rstrip(self.data_format) for s in self.slide_data['slide_id_name'].values]
		if i is not	None:
			slide_id = self.slide_data['slide_id_name'][i]
			slide_id = slide_id.split('/')[-1].rstrip(self.data_format)
			if type(self.data_dir) == dict:
				source = self.slide_data['source'][i]
				data_dir = self.data_dir[source]
			else:
				data_dir = self.data_dir
			#full_path = os.path.join(data_dir,self.slide_data['slide_id_name'][index].split('/')[-2],'{}_0_1024.npy'.format(slide_id))
			full_path = os.path.join(data_dir,'{}_0_1024.npy'.format(slide_id))
			fea = np.load(full_path, allow_pickle=True)
			features = fea[()]['feature']
			return features,slide_id


		if slide_id in new_list:
			index = new_list.index(slide_id)
			label = self.slide_data['label'][index]
			#modal = self.slide_data['modal'][index]
			#label = self.slide_data.loc[self.slide_data['slide_id_name'] == slide_id, 'label']
			#modal = self.slide_data.loc[self.slide_data['slide_id_name'] == slide_id, 'modal']
			if type(self.data_dir) == dict:
				source = self.slide_data['source'][index]
				data_dir = self.data_dir[source]
			else:
				data_dir = self.data_dir
			#full_path = os.path.join(data_dir,self.slide_data['slide_id_name'][index].split('/')[-2],'{}_0_1024.npy'.format(slide_id))
			full_path = os.path.join(data_dir,'{}_0_1024.npy'.format(slide_id))
			fea = np.load(full_path, allow_pickle=True)
			features = fea[()]['feature']
			cor = fea[()]['index']
			coords = np.array([filename.split('_')[:2] for filename in cor], dtype=int)
			TCGA_silde_id = slide_id.split('.')[0]
			caption = os.path.join('123',TCGA_silde_id+'.pt')
			if not os.path.exists(caption):
				caption_token = None
			else:
				caption_token = torch.load(caption, map_location='cpu', weights_only=True)
			gene = os.path.join('123',slide_id+'.npy')
			if os.path.exists(gene):
				rna_token = np.load(gene)
				#rna_token = torch.from_numpy(rna_token)
			elif os.path.exists(os.path.join('125',slide_id+'.pt')):
				rna_token = torch.load(os.path.join('123',slide_id+'.pt'), map_location='cpu')
			else:
				rna_token = None
			return features,caption_token,rna_token, label, coords,slide_id,self.slide_data['slide_id_name'][index].split('/')[-2]
		else:
			return None

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id_name'][idx]
		slide_id = slide_id.split('/')[-1].rstrip(self.data_format)
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				features = torch.load(full_path)
				return features, label
			else:
				return slide_id, label
		else:
			
			full_path = os.path.join(data_dir, '{}_{}.npy'.format(slide_id,self.suffix))
			
			if not os.path.exists(full_path):
				#print(data_dir)
				#TCGA-DATA
				if 'dataset' in getattr(self.slide_data, 'columns', []):
					data_name = self.slide_data.loc[idx, 'dataset']
					if data_name == 'CPTAC-CCRCC':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/cptac/CCRCC/',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'DHMC_RCC':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/DHMC_RCC/',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,'0_256'))
					elif data_name == 'DHMC_LUNG':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/DHMC_LUNG/',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'MUT':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/MUT/',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'CPTAC-LSCC':	
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/cptac/LSCC/',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'CPTAC-LUAD':	
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/cptac/LUAD/',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
				elif data_dir.split('/')[1] == 'data':
					full_path = os.path.join('/'.join(data_dir.split('/')[:-2]),data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					#print(full_path)
				elif 'type' not in getattr(self.slide_data, 'columns', []):
					type_value = self.slide_data.loc[idx, 'slide_id_name'].split('/')[6]
					full_path = os.path.join('/'.join(data_dir.split('/')[:-2]),type_value,data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					#print(full_path)
				else:
					type_value = self.slide_data.loc[idx, 'type']
					full_path = os.path.join('/'.join(data_dir.split('/')[:-2]),type_value,'clam_gen_1024',data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
			fea = np.load(full_path, allow_pickle=True)
			features = fea[()]['feature']
			cor = fea[()]['index']
			coords = np.array([filename.split('_')[:2] for filename in cor], dtype=int)
			features = torch.from_numpy(features)
			if self.return_slide_id:
				return features, label, coords,slide_id
			return features, label, coords
class Generic_WSI_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		data_format = '.svs',
		**kwargs):
		super(Generic_WSI_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False
		self.return_slide_id = False
		self.data_format = data_format
	def load_from_h5(self, toggle):
		self.use_h5 = toggle
	def return_slideid(self):
		self.return_slide_id = True
	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id_name'][idx]
		slide_id = slide_id.split('/')[-1].rstrip(self.data_format)
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				features = torch.load(full_path)
				return features, label
			else:
				return slide_id, label
		else:
			
			full_path = os.path.join(data_dir, '{}_{}.npy'.format(slide_id,self.suffix))
			if not os.path.exists(full_path):
				if 'dataset' in getattr(self.slide_data, 'columns', []):
					data_name = self.slide_data.loc[idx, 'dataset']
					if data_name == 'CPTAC-CCRCC':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/cptac/CCRCC/wsi_embedding',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'DHMC_RCC':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/DHMC_RCC/wsi_embedding',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,'0_256'))
					elif data_name == 'DHMC_LUNG':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/DHMC_LUNG/wsi_embedding',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'MUT':
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/MUT/wsi_embedding',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'CPTAC-LSCC':	
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/cptac/LSCC/wsi_embedding',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
					elif data_name == 'CPTAC-LUAD':	
						slide_id = slide_id.split('.')[0]
						full_path = os.path.join('./data/cptac/LUAD/wsi_embedding',
							   data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
				elif 'type' not in getattr(self.slide_data, 'columns', []):
					type_value = self.slide_data.loc[idx, 'slide_id_name'].split('/')[6]
					full_path = os.path.join('/'.join(data_dir.split('/')[:-3]),type_value, 'wsi_embedding', data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
				else:
					type_value = self.slide_data.loc[idx, 'type']
					full_path = os.path.join('/'.join(data_dir.split('/')[:-2]),type_value,'clam_gen_1024',data_dir.split('/')[-1], '{}_{}.npy'.format(slide_id,self.suffix))
			
			features = np.load(full_path, allow_pickle=True)
			features = torch.from_numpy(features)
			if self.return_slide_id:
				return features, label, slide_id
			return features, label
	



class Generic_caption_image_all_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		data_format = '.svs',
		**kwargs):
	
		super(Generic_MIL_all_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False
		self.return_slide_id = False
		self.data_format = data_format
	def load_from_h5(self, toggle):
		self.use_h5 = toggle
	def return_slideid(self):
		self.return_slide_id = True


	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id_name'][idx]
		slide_id = slide_id.split('/')[-1].rstrip(self.data_format)
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir,self.slide_data['slide_id_name'][idx].split('/')[-2],'{}_0_1024.npy'.format(slide_id))
				#full_path = os.path.join(data_dir,'{}.h5'.format(slide_id))
				fea = np.load(full_path, allow_pickle=True)
				features = fea[()]['feature']
				return features, label
			
			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir,self.slide_data['slide_id_name'][idx].split('/')[-2],'{}_0_1024.npy'.format(slide_id))
			#full_path = os.path.join(data_dir,'{}.h5'.format(slide_id))
			fea = np.load(full_path, allow_pickle=True)
			features = fea[()]['feature']
			cor = fea[()]['index']
			coords = np.array([filename.split('_')[:2] for filename in cor], dtype=int)

			#features = torch.from_numpy(features)
			if self.return_slide_id:
				return features, label, coords,slide_id
			return features, label, coords

	def return_splits(self, from_id=True, csv_path=None):
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_caption_all_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_caption_all_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_caption_all_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split
	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_caption_all_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)
		else:
			split = None
		
		return split


class Generic_MIL_all_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		data_format = '.svs',
		**kwargs):
	
		super(Generic_MIL_all_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False
		self.return_slide_id = False
		self.data_format = data_format
	def load_from_h5(self, toggle):
		self.use_h5 = toggle
	def return_slideid(self):
		self.return_slide_id = True


	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id_name'][idx]
		slide_id = slide_id.split('/')[-1].rstrip(self.data_format)
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir,self.slide_data['slide_id_name'][idx].split('/')[-2],'{}_0_1024.npy'.format(slide_id))
				#full_path = os.path.join(data_dir,'{}.h5'.format(slide_id))
				fea = np.load(full_path, allow_pickle=True)
				features = fea[()]['feature']
				return features, label
			
			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir,self.slide_data['slide_id_name'][idx].split('/')[-2],'{}_0_1024.npy'.format(slide_id))
			#full_path = os.path.join(data_dir,'{}.h5'.format(slide_id))
			fea = np.load(full_path, allow_pickle=True)
			features = fea[()]['feature']
			cor = fea[()]['index']
			coords = np.array([filename.split('_')[:2] for filename in cor], dtype=int)

			#features = torch.from_numpy(features)
			if self.return_slide_id:
				return features, label, coords,slide_id
			return features, label, coords

	def return_splits(self, from_id=True, csv_path=None):
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_all_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_all_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_all_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split
	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_all_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes,data_format = self.data_format)
		else:
			split = None
		
		return split




#class Generic_Split(Generic_MIL_Dataset):
#	def __init__(self, slide_data, data_dir=None, num_classes=2,data_format = '.svs',suffix = '0_1024'):
class Generic_Split:
	_cache = {}
	def __new__(cls,base,*args, **kwargs):
		parent = type(base)
		key = (cls, parent)
		if key not in cls._cache:
			name = f"{parent.__name__}_Split"
        
			dyn_cls = type(name, (cls, parent), {})
			cls._cache[key] = dyn_cls
		else:
			dyn_cls = cls._cache[key]
		self = object.__new__(dyn_cls)
		return self
	def __init__(self, base, slide_data, data_dir=None, num_classes=2,data_format = '.svs',suffix = '0_1024'):
		self.return_slide_id = False
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		self.data_format = data_format
		self.suffix = suffix
		for i in range(self.num_classes):
			class_counts = self.slide_data['label'].value_counts().to_dict()
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		

class Generic_caption_all_Split(Generic_MIL_all_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2,data_format = '.svs'):
		self.return_slide_id = False
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		self.data_format = data_format
		for i in range(self.num_classes):
			class_counts = self.slide_data['label'].value_counts().to_dict()
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
class Generic_all_Split(Generic_MIL_all_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2,data_format = '.svs'):
		self.return_slide_id = False
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		self.data_format = data_format
		for i in range(self.num_classes):
			class_counts = self.slide_data['label'].value_counts().to_dict()
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)