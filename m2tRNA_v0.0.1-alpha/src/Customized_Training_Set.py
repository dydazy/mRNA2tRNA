import pandas as pd


def Customized_Training_Set(Training_set_path, you_mRNA_data_path,Customized_Training_data_set_path):
	df = pd.read_csv(Training_set_path, index_col='gene_id')
	mRNA_select = pd.read_csv(you_mRNA_data_path, index_col="gene_id")
	tRNA = df.iloc[0:429, :]
	mRNA = df.loc[mRNA_select.index].rank()
	Customized_Training_data_set = pd.concat([tRNA, mRNA])
	Customized_Training_data_set.to_csv(Customized_Training_data_set_path)
	return Customized_Training_data_set
