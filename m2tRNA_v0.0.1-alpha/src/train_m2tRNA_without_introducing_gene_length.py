import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def train_m2tRNA(m2tRNA_net, Learning_rate, epoch_num, batch_size, train_loader, test_loader, codon_bias,
				 pearson_correlation_loss,codon_usage, trained_m2tRNA_Net_save_path):
	net1 = m2tRNA_net
	train_losses = []
	test_losses = []
	criterion1 = nn.SmoothL1Loss( )
	optimizer1 = torch.optim.Adam(net1.parameters( ), lr=Learning_rate)
	for epoch in range(epoch_num):
		train_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			# net1.train()
			inputs, labels = data
			optimizer1.zero_grad( )
			outputs = net1(codon_bias(inputs,codon_usage))
			train_loss_1 = criterion1(outputs, labels)
			train_loss_2 = criterion1(torch.sort(outputs)[1].float( ), torch.sort(labels)[1].float( ))
			train_loss_3 = pearson_correlation_loss(outputs, labels)
			loss1 = train_loss_1 * 100 + train_loss_2 + train_loss_3 * 100
			loss1.backward( )
			optimizer1.step( )
			x_1 = train_loss_1.item( )
			x_2 = train_loss_2.item( )
			x_3 = train_loss_3.item( )
			train_loss = train_loss + train_loss_1.item( ) * 100 + train_loss_2.item( ) + train_loss_3.item( ) * 100
		mean_train_loss = train_loss / (len(train_loader) * batch_size)
		train_losses.append(mean_train_loss)

		test_loss = 0
		with torch.no_grad( ):
			net1.eval( )
			for i, data in enumerate(test_loader, 0):
				test_inputs, test_labels = data
				test_pridect = net1(codon_bias(test_inputs,codon_usage))
				test_loss_1 = criterion1(test_pridect, test_labels)
				test_loss_2 = criterion1(torch.sort(test_pridect)[1].float( ), torch.sort(test_labels)[1].float( ))
				test_loss_3 = pearson_correlation_loss(test_pridect, test_labels)
				y_1 = test_loss_1.item( )
				y_2 = test_loss_2.item( )
				y_3 = test_loss_3.item( )
				test_loss = test_loss + test_loss_1.item( ) * 100 + test_loss_2.item( ) + test_loss_3.item( ) * 100
			mean_test_loss = test_loss / (len(test_loader) * batch_size)
			test_losses.append(mean_test_loss)

		if epoch % 10 == 0:
			print(
				'[epoch == %d],Train loss: %.3f, SmoothL1Loss: %.3f, SmoothL1Loss_rank: %.3f, pearson_correlation_loss(1-pearson): %.3f' % (
					epoch, mean_train_loss, x_1, x_2, x_3))
			print(
				'[epoch == %d],Test loss: %.3f,SmoothL1Loss: %.3f, SmoothL1Loss_rank: %.3f, pearson_correlation_loss(1-pearson): %.3f' % (
					epoch, mean_test_loss, y_1, y_2, y_3))

		if epoch == epoch_num - 1:
			ttloss = pd.DataFrame( )
			ttloss["train_losses"] = train_losses
			ttloss["test_losses"] = test_losses
			ttloss.to_csv(trained_m2tRNA_Net_save_path + "m2tRNA_" + str(epoch+1) + "_loss.csv")
			torch.save(net1, "m2tRNA_" + str(epoch+1) + "epoch.pkl")
			plt.plot(test_losses[0:len(test_losses) - 1], label="Test Loss", alpha=0.5)
			plt.plot(train_losses[1:len(train_losses)], label="Train Loss", alpha=1)
			plt.legend( )
			plt.savefig(trained_m2tRNA_Net_save_path+'m2tRNA_' + str(epoch+1) + '_loss.jpg', dpi=600)
			plt.close( )
	return net1
