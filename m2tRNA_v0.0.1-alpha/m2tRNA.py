import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.prepare_training_set import merge_csv_files
from src.Customized_Training_Set import Customized_Training_Set
from src.data_set_prepare import data_set_prepare_function, tRNA_codon_prepare_function, codon_usage_prepare_function, \
	activate_list_prepare_function, cds_length_prepare_function
from src.bioinfo_Integrate import codon_bias, codon_bias_length, pearson_correlation_loss
from src.m2tRNA_Net import m2tRNA
from src.train_m2tRNA_without_introducing_gene_length import train_m2tRNA
from src.train_m2tRNA_introducing_gene_length import train_m2tRNA_with_length
from src.predict_tRNA_level import get_tRNA_levels

parser = argparse.ArgumentParser(description="The deployment of m2tRNA requires the following parameters ^_^: ")

# --get_training_set
parser.add_argument('--get_training_set', type=bool, default=True,
					help='If you are not ready for training set, please set True ,"--get_training_set True"')

parser.add_argument('--Custom_training_set_path', type=bool, default=False,
					help='Please enter the absolute path where the custom gene set is saved. Please note that this parameter is only effective when the ‘–get_training_set’ parameter is\
					 set to True. This parameter sets the save path for the automatically generated training set. By \
					 default, this parameter is False, which means the default save path is “m2tRNA/Process_file/Rawdata_tRNA_mRNA.csv.”')
parser.add_argument('--split_training_set_dir', type=str, default="default",
					help='Unless the ‘--Custom_training_set_path’ is set to ‘True’, this parameter does not need to be filled in.\
					The training set we provide after splitting, the default path is ‘/m2tRNA/src/Rawdata_tRNA_mRNA’.')
parser.add_argument('--merged_training_set_dir', type=str, default="default",
					help='Unless the ‘--Custom_training_set_path’ is set to ‘True’, this parameter does not need to be filled in.\
					The training set we provide after splitting, the default path is ‘/m2tRNA/Process_file’.')

# --Use_default_path
parser.add_argument('--Use_default_path', type=bool, default=True,
					help='If True, all process files and result files will be saved to the default path. The process files will be saved in the ‘m2tRNA/Process_file’ directory. \
					The result files will be directly output to the ‘m2tRNA/’ directory.')
# --bioinfo path
parser.add_argument('--tRNA_codon_path', type=str, default="default",
					help='Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in. This \
					parameter is used to input the path of the ‘tRNA_codon.csv’ file. This \
					is a penalty matrix with a shape of (codon*anticodon, 64*429), composed of 0 and 1. It is used to kill the \
					neurons in m2tRNA that describe non-corresponding anticodon-codon. The default path for this file \
					is ‘m2tRNA/src/Data/tRNA_codon.csv’.')
parser.add_argument('--codon_usage_path', type=str, default="default",
					help='Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.This \
					parameter is used to input the path of the ‘codon_frequence.csv’ file. This is a matrix composed of genes \
					and codon usage frequencies, with a shape of 17000*64. It is used to describe the usage frequency of each \
					codon in each gene. The default path for this file is ‘m2tRNA/src/Data/codon_frequence.csv’.')
parser.add_argument('--activate_list_path', type=str, default="default",
					help='Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.\
					This parameter is used to input the path of the ‘tRNA_activate_list.csv’ file. This file will introduce\
					 a matrix of shape 1*429, used to describe the activity of 429 tRNA genes. In fact, this is also a penalty\
					  list, defined according to the description of Thornlow BP, Armstrong J, et al., used to distinguish those \
					  high-expression high-expression/low-expression tRNA genes. \
					  Reference: [Thornlow BP, Armstrong J, et al. Predicting transfer RNA gene activity from sequence and genome \
					  context. Genome Res. 2020. 30: 85-94.]  The default path for this file is ‘m2tRNA/src/Data/tRNA_activate_list.csv’.')
parser.add_argument('--cds_length_path', type=str, default="default",
					help='Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.\
					This parameter is used to input the path of the ‘cds_length.csv’ file. This file is used to describe \
					the length of the cds for each input gene, and this value is determined by the longest cds of the gene.\
					The default path for this file is ‘m2tRNA/src/Data/cds_length.csv. Please note that this parameter is \
					optional and can be used to introduce the length of the gene when calculating tRNA expression. Although \
					this helps with the biological interpretation of m2tRNA, it does not significantly improve prediction performance.')
# --process_file path
parser.add_argument('--Customized_Training_data_set_path', type=str, default="",
					help='Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.\
					The location of this parameter is used to store the customized training set generated during the process. \
					This customized training set is determined by the gene_id in your own mRNA readcount. The default storage \
					path is ‘/Process_file/Customized_Training_data_set.csv’.')
parser.add_argument('--trained_m2tRNA_Net_save_path', type=str, default="",
					help='Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.\
					The path filled in this parameter is the path where the results of the loss function are saved when training m2tRNA.')

# --Use_default_parameter
parser.add_argument('--Use_default_parameter', type=bool, default=False,
					help='If True, all parameters will use the default settings, which include device = ‘cuda’, batch_size = 1024, Learning_rate = 0.001, and epoch_num = 10.If this parameter is set to \
					False, you need to set parameters such as  "--batch_size", "--Learning_rate ", and "--epoch_num" together.')
parser.add_argument('--device', type=str, default="cuda",
					help='Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled in. This parameter refers to the\
					 device used for training m2tRNA, which can be either ‘cuda’ or ‘cpu’. However, please note that if ‘cpu’ is selected for training m2tRNA\
					  and the training set contains a large number of genes, it may result in slower operation.')
parser.add_argument('--batch_size', type=int, default=1024,
					help='Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled in.This parameter refers to \
					the batch size setting when training m2tRNA. The size of this batch size is related to your RAM. ')
parser.add_argument('--Learning_rate', type=float, default=0.001,
					help='Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled in.\
					This parameter sets the learning rate for training m2tRNA.')
parser.add_argument('--epoch_num', type=int, default=1500,
					help='Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled \
					in. This parameter sets the epoch number for training m2tRNA.')

# --you_mRNA_data_path
parser.add_argument('--you_mRNA_data_path', type=str,
					default="default",
					help='Please enter the absolute path of your mRNA readcount data. This data is a comma-separated csv file, which includes ‘gene_id’ \
					and sample numbers. If you have any doubts, please refer to the data structure provided on GitHub. I would like to remind you again, \
					please make sure that the gene names in this file exist in the training set we provide. If this parameter is set to default, \
					it will automatically read the file named ‘test_mRNA_select.csv’ in the m2tRNA directory.')

# if get_training_set == "Custom":
# 	split_training_set_dir = args.split_training_set_dir
# 	merged_training_set_dir = args.merged_training_set_dir
# 	merge_csv_files(split_training_set_dir, merged_training_set_dir)
# tRNA_codon_path = args.tRNA_codon_path
# codon_usage_path = args.codon_usage_path
# activate_list_path = args.activate_list_path
# cds_length_path = args.cds_length_path
# Training_set_path = args.Training_set_path
# Customized_Training_data_set_path = args.Customized_Training_data_set_path
# trained_m2tRNA_Net_save_path = args.trained_m2tRNA_Net_save_path

args = parser.parse_args( )
get_training_set = args.get_training_set
Use_default_path = args.Use_default_path
Use_default_parameter = args.Use_default_parameter
you_mRNA_data_path = args.you_mRNA_data_path
Custom_training_set_path = args.Custom_training_set_path
script_directory = os.path.split(os.path.realpath(__file__))[0]
# get_training_set = True
# Use_default_path = True
# Use_default_parameter = True
# you_mRNA_data_path = "D:/Z_Jupyter/tRNA/基于RNAseq预测tRNA的表达量/code availability/3.m2tRNA0.1_beta/test_mRNA_select.csv"

split_training_set_dir = script_directory + "/src/Rawdata_tRNA_mRNA"
merged_training_set_dir = script_directory + "/Process_file/Rawdata_tRNA_mRNA.csv"

if Custom_training_set_path == True:
	split_training_set_dir = args.split_training_set_dir
	merged_training_set_dir = args.merged_training_set_dir

if get_training_set == True:
	print("----------------Compiling the training set from the split files.--------------------")
	merge_csv_files(split_training_set_dir, merged_training_set_dir)

if Use_default_path == True:
	tRNA_codon_path = script_directory + "/src/Data/tRNA_codon.csv"
	codon_usage_path = script_directory + "/src/Data/codon_frequece.csv"
	activate_list_path = script_directory + "/src/Data/tRNA_activate_list.csv"
	cds_length_path = script_directory + "/src/Data/cds_length.csv"
	Training_set_path = merged_training_set_dir
	Customized_Training_data_set_path = script_directory + "/Process_file/Customized_Training_data_set.csv"
	trained_m2tRNA_Net_save_path = script_directory + "/Process_file/"
else:
	tRNA_codon_path = args.tRNA_codon_path
	codon_usage_path = args.codon_usage_path
	activate_list_path = args.activate_list_path
	cds_length_path = args.cds_length_path
	Training_set_path = args.Training_set_path
	Customized_Training_data_set_path = args.Customized_Training_data_set_path
	trained_m2tRNA_Net_save_path = args.trained_m2tRNA_Net_save_path

device = args.device
batch_size = args.batch_size
Learning_rate = args.Learning_rate
epoch_num = args.epoch_num

if Use_default_parameter == True:
	device = "cuda"
	batch_size = 1024
	Learning_rate = 0.001
	epoch_num = 10

if you_mRNA_data_path == "default":
	you_mRNA_data_path = script_directory + "/test_mRNA_select.csv"
else:
	you_mRNA_data_path = args.you_mRNA_data_path

print("----------------Generating a customized training set.--------------------")
Customized_Training_data_set = Customized_Training_Set(Training_set_path, you_mRNA_data_path,
													   Customized_Training_data_set_path)
print("DONE")

print("----------------prepare data set for m2tRNA training.--------------------")
train_loader, test_loader, feature_index, df, X_validate, y_validate = data_set_prepare_function(
	Customized_Training_data_set_path,
	device, batch_size)
print("DONE")

print("----------------Importing biological information.--------------------")
print("----------------Importing the codon usage frequency of the specified genes.--------------------")
codon_usage, codon_usage_data = codon_usage_prepare_function(codon_usage_path, df, device)
print("DONE")
print("----------------Importing biological information.--------------------")
activate_list = activate_list_prepare_function(activate_list_path, df, device)
print("DONE")
print("----------------Importing the tRNA gene activity list.--------------------")
cds_length = cds_length_prepare_function(cds_length_path, df, device)
print("DONE")
print("----------------Setting up the correspondence between codons and anticodons.--------------------")
tRNA_codon = tRNA_codon_prepare_function(tRNA_codon_path, codon_usage_data, df, device)
print("DONE")

print("----------------Initializing m2tRNA.--------------------")
m2tRNA_net = m2tRNA(df, tRNA_codon, activate_list).to(device)
print("DONE")

print("----------------Starting to train m2tRNA.--------------------")
trained_m2tRNA = train_m2tRNA(m2tRNA_net, Learning_rate, epoch_num, batch_size, train_loader, test_loader, codon_bias,
							  pearson_correlation_loss, codon_usage, trained_m2tRNA_Net_save_path)
print("----------------Starting to train m2tRNA.--------------------")
print("DONE")
# trained_m2tRNA_length = train_m2tRNA_with_length(m2tRNA_net, Learning_rate, epoch_num, batch_size, train_loader,
# 												 test_loader, codon_bias_length, pearson_correlation_loss, codon_usage,
# 												 cds_length, trained_m2tRNA_Net_save_path)
print("----------------Predicting the tRNA expression of the target samples.--------------------")
tRNA_data = get_tRNA_levels(df, codon_usage, you_mRNA_data_path, trained_m2tRNA, device)
print("----------------Saving the results.--------------------")
print("DONE")
