import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def data_set_prepare_function(Training_set_path, device, batch_size):
	df = pd.read_csv(Training_set_path, index_col='gene_id')
	X = df.iloc[429:df.shape[0], :]
	Y = df.iloc[0:429, :]
	X_train_index, X_test_validate_index, y_train_index, y_test_validate_index = train_test_split(X.columns.to_list( ),
																								  Y.columns.to_list( ),
																								  test_size=0.4,
																								  random_state=40)
	X_validate_index, X_test_index, y_validate_index, y_test_index = train_test_split(X_test_validate_index,
																					  y_test_validate_index,
																					  test_size=0.5, random_state=40)
	X = df.iloc[429:df.shape[0], :].values.T
	Y = df.iloc[0:429, :].values.T
	feature_index = df.index[429:len(df)].tolist( )
	X_train, X_test_validate, y_train, y_test_validate = train_test_split(X, Y, test_size=0.4, random_state=40)
	X_validate, X_test, y_validate, y_test = train_test_split(X_test_validate, y_test_validate, test_size=0.5,
															  random_state=40)
	X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
	y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
	X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
	y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
	X_validate = torch.tensor(X_validate, dtype=torch.float32).to(device)
	y_validate = torch.tensor(y_validate, dtype=torch.float32).to(device)
	train_data = TensorDataset(X_train, y_train)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_data = TensorDataset(X_test, y_test)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
	return train_loader, test_loader, feature_index, df, X_validate, y_validate


def codon_usage_prepare_function(codon_usage_path, df, device):
	codon_usage_data = pd.read_csv(codon_usage_path, index_col='gene_id')
	codon_usage = codon_usage_data.loc[df.iloc[429:df.shape[0], 0:1].index]
	codon_usage = torch.from_numpy(codon_usage.values).to(device)
	codon_usage = codon_usage.to(torch.float32)
	return codon_usage, codon_usage_data


def activate_list_prepare_function(activate_list_path, df, device):
	activate_list_data = pd.read_csv(activate_list_path, index_col='gene_id')
	activate_list = activate_list_data.loc[df.iloc[0:429, 0:1].index]
	activate_list = torch.from_numpy(activate_list.values).to(device)
	activate_list = activate_list.to(torch.float32)
	return activate_list


def cds_length_prepare_function(cds_length_path, df, device):
	cds_length_data = pd.read_csv(cds_length_path, index_col='gene_id')
	cds_length = cds_length_data.loc[df.iloc[429:df.shape[0], 0:1].index]
	cds_length = torch.from_numpy(cds_length.values).to(device)
	cds_length = cds_length.to(torch.float32)
	return cds_length


def tRNA_codon_prepare_function(tRNA_codon_path, codon_usage_data, df, device):
	tRNA_codon_data = pd.read_csv(tRNA_codon_path, index_col='gene_id')
	tRNA_codon = tRNA_codon_data.loc[df.iloc[0:429, 0:1].index]
	tRNA_codon = tRNA_codon.T.loc[codon_usage_data.T.index].T
	tRNA_codon = torch.from_numpy(tRNA_codon.values).to(device)
	tRNA_codon = tRNA_codon.to(torch.float32).T
	return tRNA_codon
