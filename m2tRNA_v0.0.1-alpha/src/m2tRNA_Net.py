import torch.nn as nn
import torch.nn.functional as F
import torch


class m2tRNA(nn.Module):
	def __init__(self, df, tRNA_codon, activate_list):
		super(m2tRNA, self).__init__( )
		self.fc1 = nn.Linear(df.shape[0] - 429, 429)
		self.fc5 = nn.Linear(64, 429)
		self.act = nn.LeakyReLU(negative_slope=0.01)
		self.tRNA_codon = tRNA_codon
		self.activate_list = activate_list

	def forward(self, x):
		x1 = self.fc1(torch.transpose(x, 1, 2))
		x_cat = self.act((x1))
		x_cat = x_cat * (self.tRNA_codon)
		x_cat = self.act(self.fc5(torch.transpose(x_cat, 1, 2)))
		x_cat = x_cat * (self.activate_list)
		x7 = F.relu(self.act(x_cat.mean(2)))
		x7 = torch.squeeze(x7)
		return x7
