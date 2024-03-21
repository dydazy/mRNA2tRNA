import pandas as pd
import torch
import numpy as np
from src.bioinfo_Integrate import codon_bias

def get_tRNA_levels(df, codon_usage, you_mRNA_data_path, model,device):
	model = model.to(device)
	codon_usage = codon_usage.to(device)
	mrnadata = pd.read_csv(you_mRNA_data_path, index_col="gene_id").rank()
	mrnadata_tensor = torch.tensor(np.array(mrnadata), dtype=torch.float32).to(device).T
	predicted_tRNA_level = model(codon_bias(mrnadata_tensor, codon_usage)).to(device)
	predicted_tRNA_level_cpu=predicted_tRNA_level.to("cpu")
	tRNA_data = pd.DataFrame(predicted_tRNA_level_cpu.detach( ).numpy( ).T)
	tRNA_data["gene_id"] = df.iloc[0:429, 0:1].index
	tRNA_data = tRNA_data.set_index("gene_id")
	tRNA_data.columns = mrnadata.columns.tolist( )
	tRNA_data.to_csv('predicted_tRNA_level.csv', index=True)
	return tRNA_data
