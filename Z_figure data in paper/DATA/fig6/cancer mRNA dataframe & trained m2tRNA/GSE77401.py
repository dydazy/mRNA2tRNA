import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from shapash.data.data_loader import data_loading
from sklearn.metrics import roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_str="GSE77401"
##select data
df = pd.read_csv(path_str+'/MDA_DE_log_tRNA_rank_mRNA.csv',index_col='gene_id')


X = df.iloc[429:df.shape[0],:]
Y = df.iloc[0:429, :]
X_train_index, X_test_validate_index, y_train_index, y_test_validate_index = train_test_split(X.columns.to_list(), Y.columns.to_list(), test_size=0.2, random_state=40)
X_validate_index, X_test_index, y_validate_index, y_test_index = train_test_split(X_test_validate_index, y_test_validate_index, test_size = 0.5, random_state = 40)

X = df.iloc[429:df.shape[0],:].values.T
Y = df.iloc[0:429, :].values.T

X_train, X_test_validate, y_train, y_test_validate = train_test_split(X, Y, test_size=0.4, random_state=40)
X_validate, X_test, y_validate, y_test = train_test_split(X_test_validate, y_test_validate, test_size = 0.5, random_state = 40)
# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
#X_validate=torch.tensor(X_validate, dtype=torch.float32).to(device)
#y_validate=torch.tensor(y_validate, dtype=torch.float32).to(device)
df.shape[0]-429

batch_size=2048
# Create data loaders
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_validate.shape)
print(y_validate.shape)
len(train_loader)

##codonbias data
codon_usage_data=pd.read_csv("Data/codon_frequece.csv",index_col='gene_id')
codon_usage=codon_usage_data.loc[df.iloc[429:df.shape[0], 0:1].index]
codon_usage = torch.from_numpy(codon_usage.values).to(device)
codon_usage=codon_usage.to(torch.float32)

##activate data
activate_list_data=pd.read_csv("Data/tRNA_activate_list.csv",index_col='gene_id')
activate_list=activate_list_data.loc[df.iloc[0:429, 0:1].index]
activate_list = torch.from_numpy(activate_list.values).to(device)
activate_list=activate_list.to(torch.float32)

##cdslength data
cds_length_data=pd.read_csv("Data/cds_length.csv",index_col='gene_id')
cds_length=cds_length_data.loc[df.iloc[429:df.shape[0], 0:1].index]
cds_length = torch.from_numpy(cds_length.values).to(device)
cds_length=cds_length.to(torch.float32)

##codon_contribute
tRNA_codon_data=pd.read_csv("Data/tRNA_codon.csv",index_col='gene_id')
tRNA_codon=tRNA_codon_data.loc[df.iloc[0:429, 0:1].index]
tRNA_codon=tRNA_codon.T.loc[codon_usage_data.T.index].T
tRNA_codon = torch.from_numpy(tRNA_codon.values).to(device)
tRNA_codon=tRNA_codon.to(torch.float32).T

print(codon_usage_data.shape)
print(codon_usage.shape)
print(activate_list_data.shape)
print(activate_list.shape)
print(cds_length_data.shape)
print(cds_length.shape)

##codon_frequence
def codon_bias(read_counts):
    if read_counts.dim()==1:
        read_counts=read_counts.flatten().unsqueeze(0)
    x=torch.einsum('ij,jk->ijk',read_counts,codon_usage)
    return x
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

##cds_length,codon_frequence
def codon_bias_length(read_counts):
    if read_counts.dim()==1:
        seq_pool=read_counts.flatten()*cds_length.flatten()
        seq_pool=seq_pool.unsqueeze(0)
        x=torch.einsum('ij,jk->ijk',seq_pool,codon_usage)
    else:
        seq_pool=torch.einsum('ij,jk->ijk', cds_length.T,codon_usage).squeeze()
        x=torch.einsum('ij,jk->ijk',read_counts,seq_pool)
    return x
#optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

###pearson_correlation_loss
def pearson_correlation_loss(x, y):
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)
    x_std = torch.std(x, dim=1, keepdim=True)
    y_std = torch.std(y, dim=1, keepdim=True)
    n = x.size(1)
    vx = x - x_mean
    vy = y - y_mean
    loss = 1-torch.mean(torch.sum(vx * vy, dim=1) / (n * x_std * y_std))
    return loss

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        #self.my_module = codon_bias()
        self.fc1 = nn.Linear(df.shape[0]-429, 429) 
        self.fc5 = nn.Linear(64, 429)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        ##
        
    def forward(self, x):
        x1 = self.fc1(torch.transpose(x, 1, 2))
        x_cat = self.act((x1))
        x_cat=x_cat*tRNA_codon
        x_cat = self.act(self.fc5(torch.transpose(x_cat, 1, 2))) 
        x_cat = x_cat*activate_list
        x7=F.relu(self.act(x_cat.mean(2)))
        x7=torch.squeeze(x7)
        
        return x7
    
net1 = Net1().to(device)
train_losses=[]
test_losses=[]
 

criterion1 = nn.SmoothL1Loss()
optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
for epoch in range(1001):
    train_loss = 0.0   
    for i, data in enumerate(train_loader, 0):
        #net1.train()
        inputs, labels = data
        optimizer1.zero_grad()
        outputs = net1(codon_bias(inputs))
        train_loss_1 = criterion1(outputs, labels)
        train_loss_2 = criterion1(torch.sort(outputs)[1].float(), torch.sort(labels)[1].float())
        train_loss_3 = pearson_correlation_loss(outputs, labels)
        loss1=train_loss_1*1000+train_loss_2
        loss1.backward()
        optimizer1.step()
        x=train_loss_1.item()
        train_loss =train_loss+ train_loss_1.item()*1000+train_loss_2.item()
    mean_train_loss =train_loss / (len(train_loader) * batch_size)
    train_losses.append(mean_train_loss)
    
    test_loss=0
    with torch.no_grad():
        net1.eval()
        for i, data in enumerate(test_loader, 0):
            test_inputs, test_labels = data
            test_pridect = net1(codon_bias(test_inputs))
            test_loss_1 = criterion1(test_pridect, test_labels)
            test_loss_2 = criterion1(torch.sort(test_pridect)[1].float(), torch.sort(test_labels)[1].float())
            test_loss_3 = pearson_correlation_loss(test_pridect, test_labels)
            test_loss = test_loss+ test_loss_1.item()*1000+ test_loss_2.item()
            y=test_loss_1.item()
        mean_test_loss=test_loss / (len(test_loader) * batch_size)
        test_losses.append(mean_test_loss)
    
    if epoch%20 ==0:
        print(path_str+'_MDA_DE %d],Train loss: %.3f,train_loss_1: %.3f, train_loss_2: %.3f, train_loss_3: %.3f' % (epoch,mean_train_loss,x,train_loss_2,train_loss_3))
        print(path_str+'_MDA_DE %d],Test loss: %.3f,test_loss_1: %.3f, test_loss_2: %.3f, test_loss_3: %.3f' % (epoch,mean_test_loss,y,test_loss_2,test_loss_3))
        
    if epoch%100 ==0:
        ttloss=pd.DataFrame()
        ttloss["train_losses"]=train_losses
        ttloss["test_losses"]=test_losses
        ttloss.to_csv(path_str+'/MDA_DE_'+str(epoch)+'_loss.csv')
        torch.save(net1, path_str+'/MDA_DE_'+str(epoch)+'_loss.pkl')
        plt.plot(test_losses[0:len(test_losses)-1], label="Test Loss",alpha=0.5)
        plt.plot(train_losses[1:len(train_losses)], label="Train Loss",alpha=1)
        plt.legend()
        plt.savefig(path_str+'/MDA_DE_'+str(epoch)+'_loss.jpg', dpi=600)
        plt.close()
        
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from shapash.data.data_loader import data_loading
from sklearn.metrics import roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##select data
df = pd.read_csv(path_str+'/CN34_DE_log_tRNA_rank_mRNA.csv',index_col='gene_id')

###########获取数据引索########
X = df.iloc[429:df.shape[0],:]
Y = df.iloc[0:429, :]
X_train_index, X_test_validate_index, y_train_index, y_test_validate_index = train_test_split(X.columns.to_list(), Y.columns.to_list(), test_size=0.2, random_state=40)
X_validate_index, X_test_index, y_validate_index, y_test_index = train_test_split(X_test_validate_index, y_test_validate_index, test_size = 0.5, random_state = 40)

X = df.iloc[429:df.shape[0],:].values.T
Y = df.iloc[0:429, :].values.T
#20%测试集 20%验证集 60%训练集
X_train, X_test_validate, y_train, y_test_validate = train_test_split(X, Y, test_size=0.4, random_state=40)
X_validate, X_test, y_validate, y_test = train_test_split(X_test_validate, y_test_validate, test_size = 0.5, random_state = 40)
# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
#X_validate=torch.tensor(X_validate, dtype=torch.float32).to(device)
#y_validate=torch.tensor(y_validate, dtype=torch.float32).to(device)
df.shape[0]-429

batch_size=2048
# Create data loaders
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_validate.shape)
print(y_validate.shape)
len(train_loader)

##codonbias data
codon_usage_data=pd.read_csv("Data/codon_frequece.csv",index_col='gene_id')
codon_usage=codon_usage_data.loc[df.iloc[429:df.shape[0], 0:1].index]
codon_usage = torch.from_numpy(codon_usage.values).to(device)
codon_usage=codon_usage.to(torch.float32)

##activate data
activate_list_data=pd.read_csv("Data/tRNA_activate_list.csv",index_col='gene_id')
activate_list=activate_list_data.loc[df.iloc[0:429, 0:1].index]
activate_list = torch.from_numpy(activate_list.values).to(device)
activate_list=activate_list.to(torch.float32)

##cdslength data
cds_length_data=pd.read_csv("Data/cds_length.csv",index_col='gene_id')
cds_length=cds_length_data.loc[df.iloc[429:df.shape[0], 0:1].index]
cds_length = torch.from_numpy(cds_length.values).to(device)
cds_length=cds_length.to(torch.float32)

##codon_contribute
tRNA_codon_data=pd.read_csv("Data/tRNA_codon.csv",index_col='gene_id')
tRNA_codon=tRNA_codon_data.loc[df.iloc[0:429, 0:1].index]
tRNA_codon=tRNA_codon.T.loc[codon_usage_data.T.index].T
tRNA_codon = torch.from_numpy(tRNA_codon.values).to(device)
tRNA_codon=tRNA_codon.to(torch.float32).T

print(codon_usage_data.shape)
print(codon_usage.shape)
print(activate_list_data.shape)
print(activate_list.shape)
print(cds_length_data.shape)
print(cds_length.shape)

##codon_frequence
def codon_bias(read_counts):
    if read_counts.dim()==1:
        read_counts=read_counts.flatten().unsqueeze(0)
    x=torch.einsum('ij,jk->ijk',read_counts,codon_usage)
    return x
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

##cds_length,codon_frequence
def codon_bias_length(read_counts):
    if read_counts.dim()==1:
        seq_pool=read_counts.flatten()*cds_length.flatten()
        seq_pool=seq_pool.unsqueeze(0)
        x=torch.einsum('ij,jk->ijk',seq_pool,codon_usage)
    else:
        seq_pool=torch.einsum('ij,jk->ijk', cds_length.T,codon_usage).squeeze()
        x=torch.einsum('ij,jk->ijk',read_counts,seq_pool)
    return x
#optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

###pearson_correlation_loss
def pearson_correlation_loss(x, y):
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)
    x_std = torch.std(x, dim=1, keepdim=True)
    y_std = torch.std(y, dim=1, keepdim=True)
    n = x.size(1)
    vx = x - x_mean
    vy = y - y_mean
    loss = 1-torch.mean(torch.sum(vx * vy, dim=1) / (n * x_std * y_std))
    return loss

##q1Net1
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        #self.my_module = codon_bias()
        self.fc1 = nn.Linear(df.shape[0]-429, 429) 
        self.fc5 = nn.Linear(64, 429)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        ##
        
    def forward(self, x):
        x1 = self.fc1(torch.transpose(x, 1, 2))
        x_cat = self.act((x1))
        x_cat=x_cat*tRNA_codon
        x_cat = self.act(self.fc5(torch.transpose(x_cat, 1, 2))) 
        x_cat = x_cat*activate_list
        x7=F.relu(self.act(x_cat.mean(2)))
        x7=torch.squeeze(x7)
        
        return x7
    
net1 = Net1().to(device)
train_losses=[]
test_losses=[]
 

criterion1 = nn.SmoothL1Loss()
optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
for epoch in range(1001):
    train_loss = 0.0   
    for i, data in enumerate(train_loader, 0):
        #net1.train()
        inputs, labels = data
        optimizer1.zero_grad()
        outputs = net1(codon_bias(inputs))
        train_loss_1 = criterion1(outputs, labels)
        train_loss_2 = criterion1(torch.sort(outputs)[1].float(), torch.sort(labels)[1].float())
        train_loss_3 = pearson_correlation_loss(outputs, labels)
        loss1=train_loss_1*1000+train_loss_2
        loss1.backward()
        optimizer1.step()
        x=train_loss_1.item()
        train_loss =train_loss+ train_loss_1.item()*1000+train_loss_2.item()
    mean_train_loss =train_loss / (len(train_loader) * batch_size)
    train_losses.append(mean_train_loss)
    
    test_loss=0
    with torch.no_grad():
        net1.eval()
        for i, data in enumerate(test_loader, 0):
            test_inputs, test_labels = data
            test_pridect = net1(codon_bias(test_inputs))
            test_loss_1 = criterion1(test_pridect, test_labels)
            test_loss_2 = criterion1(torch.sort(test_pridect)[1].float(), torch.sort(test_labels)[1].float())
            test_loss_3 = pearson_correlation_loss(test_pridect, test_labels)
            test_loss = test_loss+ test_loss_1.item()*1000+ test_loss_2.item()
            y=test_loss_1.item()
        mean_test_loss=test_loss / (len(test_loader) * batch_size)
        test_losses.append(mean_test_loss)
    
    if epoch%20 ==0:
        print(path_str+'_CN34_DE %d],Train loss: %.3f,train_loss_1: %.3f, train_loss_2: %.3f, train_loss_3: %.3f' % (epoch,mean_train_loss,x,train_loss_2,train_loss_3))
        print(path_str+'_CN34_DE %d],Test loss: %.3f,test_loss_1: %.3f, test_loss_2: %.3f, test_loss_3: %.3f' % (epoch,mean_test_loss,y,test_loss_2,test_loss_3))
        
    if epoch%100 ==0:
        ttloss=pd.DataFrame()
        ttloss["train_losses"]=train_losses
        ttloss["test_losses"]=test_losses
        ttloss.to_csv(path_str+'/CN34_DE_'+str(epoch)+'_loss.csv')
        torch.save(net1, path_str+'/CN34_DE_'+str(epoch)+'_loss.pkl')
        plt.plot(test_losses[0:len(test_losses)-1], label="Test Loss",alpha=0.5)
        plt.plot(train_losses[1:len(train_losses)], label="Train Loss",alpha=1)
        plt.legend()
        plt.savefig(path_str+'/CN34_DE_'+str(epoch)+'_loss.jpg', dpi=600)
        plt.close()