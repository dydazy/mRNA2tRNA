import torch




def codon_bias(read_counts,codon_usage):
	if read_counts.dim( ) == 1:
		read_counts = read_counts.flatten( ).unsqueeze(0)
	x = torch.einsum('ij,jk->ijk', read_counts, codon_usage)
	return x


def codon_bias_length(read_counts,codon_usage,cds_length):
	if read_counts.dim( ) == 1:
		seq_pool = read_counts.flatten( ) * cds_length.flatten( )
		seq_pool = seq_pool.unsqueeze(0)
		x = torch.einsum('ij,jk->ijk', seq_pool, codon_usage)
	else:
		seq_pool = torch.einsum('ij,jk->ijk', cds_length.T, codon_usage).squeeze( )
		x = torch.einsum('ij,jk->ijk', read_counts, seq_pool)
	return x


def pearson_correlation_loss(x, y):
	x_mean = torch.mean(x, dim=1, keepdim=True)
	y_mean = torch.mean(y, dim=1, keepdim=True)
	x_std = torch.std(x, dim=1, keepdim=True)
	y_std = torch.std(y, dim=1, keepdim=True)
	n = x.size(1)
	vx = x - x_mean
	vy = y - y_mean
	loss = 1 - torch.mean(torch.sum(vx * vy, dim=1) / (n * x_std * y_std))
	return loss
