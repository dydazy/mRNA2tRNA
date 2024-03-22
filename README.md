# 1.Introduction of m2tRNA
m2tRNA is a tool that uses a machine learning model to convert genome-wide mRNA data into tRNA levels. This model includes biological constraints to make accurate predictions.  


## 1.1 m2tRNA framework  
![fig1-4 0](https://github.com/dydazy/mRNA2tRNA/assets/46813403/81224c04-a128-4380-8ed6-1e9174495a9c)  
fig1.Overview of the m2tRNA framework.    
a, Description of data sources, which includes expression data from the TCGA database (a) and tRNA data from DBtRend (g) (Lee, J. O., Lee, M. & Chung, Y. J. DBtRend: A Web-Server of tRNA Expression Profiles from Small RNA Sequencing Data in Humans. Genes (Basel) 12 (2021). https://doi.org:10.3390/genes12101576).  b, Methods for data preprocessing, with the recommended method indicated by a red star.  c, Transformation of input data, where mRNA expression information is weighted according to the codon usage per gene.  d, The weighted data is then input into the first layer of a fully connected neural network. To reduce computational load, a group or multiple groups of genes of interest can be selected for pre-training m2tRNA.  e, The information obtained in (d) is integrated, then constrained using codons of different amino acids.  f, It is then weighted by previously reported tRNA activity to obtain the trained m2tRNA results.  g, The DBtRend database provides tRNA profiles to validate the results from m2tRNA.  

m2tRNA utilizes machine learning to predict tRNA levels from mRNA data, integrating biological insights (Fig.1, more details in METHODS).  To adapt m2tRNA to mRNA data from various sources, ranked mRNA reads were used as input (Fig.1 b).  Considering the greater number of mRNAs than tRNAs, we employed a Multilayer Perceptron (MLP) for effective dimension transformation.  We then weighted the codon preference information onto the mRNA data (Fig.1 c) as the input for the first layer of the MLP (Fig.1 d).  The first layer output matrix was constrained by amino acid-based information and input into the second layer of MLP (Fig.1 e).  The results were weighted using tRNA activity (Fig.1 f), where tRNA activity was defined as a set of weights based on previous research to describe tRNA levels (17).  Finally, we integrated the information of the codons corresponding to each tRNA gene to obtain the predicted tRNA levels.   


## 1.2 Introduction of this Git repository  
We have made the m2tRNA available for testing on this Git repository, and it is compatible with JupyterLab/Jupyter Notebook. Additionally, we have included the replication process for all data presented in the manuscript.  


### Here we provide explanations for the content and purpose of all the files in this git repository.  
### 1.2.1 dir: ```mRNA2tRNA/1.Prepare data```  

```mRNA2tRNA/1.Prepare data/Rawdata_tRNA_mRNA``` The training set, splited, is provided in this folder.This dataset encompasses approximately 10,000 sample sets. The mRNA data within these sets is sourced from the TCGA database, while the tRNA levels are derived from the tRend database, ensuring a correspondence between the samples (as depicted in Fig1.a&g).  

```mRNA2tRNA/Prepare data.ipynb``` This file provides the process of splitting the original data and the process of merging the split data, both of which can be replicated in JupyterLab/Jupyter Notebook.  
In detail:  Prior to initiating the use of m2tRNA, it is necessary to first prepare the preliminary dataset. This dataset is primarily utilized for training within the m2tRNA neural network. We furnish all the required datasets for training m2tRNA in the section titled.  
We recommend running the ```Prepare data.ipynb``` in JupyterLab/Jupyter notebook. We provide scripts in this file to prepare the dataset needed for training. After running, you will obtain a large matrix (filename: ```tRNA_log2_mRNA_rank.csv``` or ```tRNA_log2_norm_mRNA_log2_norm.csv```,The file size is approximately ```2,700 MB```.), where rows and columns represent genes and samples respectively. The first 429 rows of the synthesized data frame consist of tRNA expression data, and the subsequent data is mRNA data.  


### 1.2.2 dir: ```mRNA2tRNA/2.Model```  

```mRNA2tRNA/2.Model/Data/``` This file provides the biological information that needs to be imported when running m2tRNA as well as containing biological information related to m2tRNA, such as codon usage frequency, tRNA gene activity, and CDS length. We recommend running these directly on JupyterLab/Jupyter Notebook.  
In detail:  We provide the m2tRNA for testing and other necessary files for its operation in the ```2.Model``` folder. This includes the main body of m2tRNA in   

```mRNA2tRNA/2.Model/2.Model.ipynb``` This file provides the basic components of m2tRNA, including the import of training data, the processing procedure, the code of m2tRNA, and the training process of m2tRNA.  
In detail: It’s worth noting that the training set for this test version of m2tRNA is composed by default of ~17,000 genes to infer tRNA levels. If you wish to test your own samples, I suggest you organize the training set and only select those genes that are detected in your samples for training.  


### 1.2.3 dir: ```mRNA2tRNA/Z_figure data in paper/```  

```mRNA2tRNA/Z_figure data in paper/``` This folder provides the process for replicating the hands-on content.  
In detail: In order to ensure data continuity, we have made every effort to provide the process for reproducing the results mentioned in the manuscript. However, we have not provided some of the larger and less important intermediate process files.  

### 1.2.4 dir: ```mRNA2tRNA/m2tRNA_v0.0.1-alpha```  

In the ```mRNA2tRNA/m2tRNA_v0.0.1-alpha``` folder, we provide the alpha version of m2tRNA and all the necessary files for its operation (including the repeatedly provided split training set). You can run it in a Windows/Linux environment with Python installed (recommended to run on Linux). The specific structure and usage of ```m2tRNA_v0.0.1-alpha``` are in sections ```1.3``` and ```1.4```.  


## 1.3 Introduction of ```m2tRNA_v0.0.1-alpha```  

### 1.3.1 Structure of ```m2tRNA_v0.0.1-alpha```  
![m2tRNA_beta](https://github.com/dydazy/mRNA2tRNA/assets/46813403/68669657-c5fe-4a4d-a50b-e1df0c4b2120)  
fig2.The process of ```m2tRNA_v0.0.1-alpha```.  


### 1.3.2 Module and Function  

1  ```Prepare Training Set```   
```mRNA2tRNA/m2tRNA_v0.0.1-alpha/src/prepare_training_set.py```  
The function ```merge_csv_files()``` is included.   
If you have already downloaded our provided training set, or you have prepared your own training set, then this module is optional. The function ```merge_csv_files()``` accepts the parameters ```--split_training_set_dir``` and ```--merged_training_set_dir```. It concatenates our provided training set with the default path ```mRNA2tRNA/m2tRNA_v0.0.1-alpha/src/Rawdata_tRNA_mRNA``` (which is consistent with the content of ```mRNA2tRNA/1.Prepare data/Rawdata_tRNA_mRNA```) and saves it to the ```merged training set path``` directory.  


2 ```Customized Training Set```   
``` mRNA2tRNA/m2tRNA_v0.0.1-alpha/src/Customized_Training_Set.py```   
The function ```Customized_Training_Set( )``` is included.   
This function generates a customized training set based on the genes of the sample you want to predict. This function requires three parameters: ```--Training_set_path``` (which is the merged training set path in the Prepare Training Set module), ```--you_mRNA_data_path``` (the path of the sample you want to predict, the format refers to our provided test file ```mRNA2tRNA/m2tRNA_v0.0.1-alpha/test_mRNA_select.csv```, it should be noted that the gene id you provide in your sample must exist in our training set, although our training set already contains almost all protein-coding genes), and ```--Customized_Training_data_set_path``` (set the location of the generated customized training set).  


3 ```Data Set Prepare```  
```mRNA2tRNA/m2tRNA_v0.0.1-alpha/src/data_set_prepare.py```  
The function ```data_set_prepare_function( )```，```codon_usage_prepare_function( )```，```activate_list_prepare_function( )```,```cds_length_prepare_function( )```,```tRNA_codon_prepare_function()``` is included.   
This module contains several functions, all of which are used to prepare the data needed for training m2tRNA.
The function ```data_set_prepare_function()``` returns the split training set and the gene list of this training set. This function accepts three parameters, including the absolute path of the training set ```--Training_set_path```, which is usually consistent with ```--Customized_Training_data_set_path```; ```--device``` and ```--batch_size``` set the device and batch size used when training m2tRNA.  
The functions ```codon_usage_prepare_function()```, ```activate_list_prepare_function()```, ```cds_length_prepare_function()```, and ```tRNA_codon_prepare_function()``` integrate the biological information input into m2tRNA and return the corresponding tensor.  


4 ```m2tRNA Net```  
```mRNA2tRNA/m2tRNA_v0.0.1-alpha/src/m2tRNA_Net.py```  
It is m2tRNA, the main structure is shown in fig1.d-f.  
The ```m2tRNA()``` function includes three parameters that need to be passed in: ```--df``` for the training set; ```--tRNA_codon``` for the codon-anticodon correspondence; and ```--activate_list``` for the activity list of tRNA genes.  


5 ```Bioinfo Integrate```  
```mRNA2tRNA/m2tRNA_v0.0.1-alpha/src/bioinfo_Integrate.py```  
The function ```codon_bias( )```，```codon_bias_length( )```，```pearson_correlation_loss( )``` is included.   
The functions ```codon_bias()``` and ```codon_bias_length()``` essentially perform the same function，Both of them weight the input transcriptome data using the frequency of codons. The only difference is that ```codon_bias_length()``` incorporates the length of the coding sequence (CDS). These two functions accept three input parameters.(```read_counts```,```codon_usage```,```cds_length```).```pearson_correlation_loss( )```,it is a loss function that we define based on the Pearson correlation coefficient.  


6 ```train m2tRNA```  
```mRNA2tRNA/m2tRNA_v0.0.1-alpha/src/train_m2tRNA.py```  
The function ```train_m2tRNA_with/without_length( )``` is included.   
Integrate all previous information and train on m2tRNA. This will ultimately return a trained neural network, which is saved to the current directory by default. At the same time, this function will be run by default to obtain the predicted tRNA expression of the target sample.  



# 2 Running m2tRNA_v0.0.1-alpha   

## 2.1 Run m2tRNA  

First, make sure you have installed the following Python packagesor run:  
```
pip install pandas, numpy, scipy, matplotlib, sklearn, torch
```  

You can directly download the ```m2tRNA_v0.0.1-alpha``` folder. Please note that this file requires about ```3000MB``` of space. Then run the following code:  
```             
cd m2tRNA_v0.0.1-alpha
bash m2tRNA.sh
```  

Or run in the conda environment:  
```
python m2tRNA.py --get_training_set True --Custom_training_set_path False --split_training_set_dir "default" 
        --merged_training_set_dir "default" --Use_default_path True --tRNA_codon_path "default" --codon_usage_path "default" 
        --activate_list_path "default" --cds_length_path "default" --Customized_Training_data_set_path "default" 
        --trained_m2tRNA_Net_save_path "default" --Use_default_parameter False --device "cuda" --batch_size 1024 
        --Learning_rate 0.001 --epoch_num 50 --you_mRNA_data_path "default"
```  

Or run in the conda environment:  
```
python m2tRNA.py --get_training_set True --Custom_training_set_path False --split_training_set_dir "default" 
        --merged_training_set_dir "default" --Use_default_path True --tRNA_codon_path "default" --codon_usage_path "default" 
        --activate_list_path "default" --cds_length_path "default" --Customized_Training_data_set_path "default" 
        --trained_m2tRNA_Net_save_path "default" --Use_default_parameter False --device "cuda" --batch_size 1024 
        --Learning_rate 0.001 --epoch_num 50 --you_mRNA_data_path "default"
```  


## 2.2 Detail/explanation of each parameter:  

```'--get_training_set'```  
Notes'If you are not ready for training set, please set True ,"--get_training_set True"'    

```'--Custom_training_set_path'```  
Notes'Please enter the absolute path where the custom gene set is saved. Please note that
this parameter is only effective when the ‘–get_training_set’ parameter is set to True.
This parameter sets the save path for the automatically generated training set. By default,
this parameter is False, which means the default save path is “m2tRNA/Process_file/Rawdata_tRNA_mRNA.csv.”'  

```'--split_training_set_dir'```  
Notes:  'Unless the ‘--Custom_training_set_path’ is set to ‘True’, this parameter does not
need to be filled in. The training set we provide after splitting, the default path is ‘/m2tRNA/src/Rawdata_tRNA_mRNA’.'  

```'--merged_training_set_dir'```  
Notes'Unless the ‘--Custom_training_set_path’ is set to ‘True’, this parameter does not need
to be filled in.The training set we provide after splitting, the default path is ‘/m2tRNA/Process_file’.'    

```'--Use_default_path'```  
Notes'If True, all process files and result files will be saved to the default path.
The process files will be saved in the ‘m2tRNA/Process_file’ directory. The result
files will be directly output to the ‘m2tRNA/’ directory.'  

```'--tRNA_codon_path', default="default"```  
Notes'Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need
to be filled in. This parameter is used to input the path of the ‘tRNA_codon.csv’ file.
This is a penalty matrix with a shape of (codon*anticodon, 64*429, composed of 0 and 1.
It is used to kill the neurons in m2tRNA that describe non-corresponding anticodon-codon.
The default path for this file is ‘m2tRNA/src/Data/tRNA_codon.csv’.'    

```'--codon_usage_path', default="default"```    
Notes'Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to
be filled in.This parameter is used to input the path of the ‘codon_frequence.csv’ file.
This is a matrix composed of genes and codon usage frequencies, with a shape of 17000*64.
It is used to describe the usage frequency of each codon in each gene. The default path
for this file is ‘m2tRNA/src/Data/codon_frequence.csv’.'    

```'--activate_list_path', default="default"```    
Notes'Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.
This parameter is used to input the path of the ‘tRNA_activate_list.csv’ file. This file will introduce
a matrix of shape 1*429, used to describe the activity of 429 tRNA genes. In fact, this is also a penalty
list, defined according to the description of Thornlow BP, Armstrong J, et al., used to distinguish those
high-expression high-expression/low-expression tRNA genes.    
Reference: [Thornlow BP, Armstrong J, et al. Predicting transfer RNA gene activity from sequence and genome
context. Genome Res. 2020. 30: 85-94.]  The default path for this file is ‘m2tRNA/src/Data/tRNA_activate_list.csv’.'  

```'--cds_length_path', default="default"```    
Notes'Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.
This parameter is used to input the path of the ‘cds_length.csv’ file. This file is used to describe 
the length of the cds for each input gene, and this value is determined by the longest cds of the gene.
The default path for this file is ‘m2tRNA/src/Data/cds_length.csv. Please note that this parameter is 
optional and can be used to introduce the length of the gene when calculating tRNA expression. Although 
this helps with the biological interpretation of m2tRNA, it does not significantly improve prediction performance.'  

```'--Customized_Training_data_set_path', default=""```    
Notes'Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.
The location of this parameter is used to store the customized training set generated during the process. 
This customized training set is determined by the gene_id in your own mRNA readcount. The default storage 
path is ‘/Process_file/Customized_Training_data_set.csv’.'  

```'--trained_m2tRNA_Net_save_path', default=""```  
Notes'Unless the ‘--Use_default_path’ is set to ‘False’, this parameter does not need to be filled in.
The path filled in this parameter is the path where the results of the loss function are saved when training m2tRNA.'  

```'--Use_default_parameter', default=False,```  
Notes'If True, all parameters will use the default settings, which include device = ‘cuda’, batch_size = 1024,
Learning_rate = 0.001, and epoch_num = 10.If this parameter is set to 
False, you need to set parameters such as  "--batch_size", "--Learning_rate ", and "--epoch_num" together.'  

```'--device', default="cuda"```  
Notes'Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled in.
This parameter refers to the device used for training m2tRNA, which can be either ‘cuda’ or ‘cpu’.
However, please note that if ‘cpu’ is selected for training m2tRNA and the training set contains
a large number of genes, it may result in slower operation.'  

```'--batch_size', type=int, default=1024```  
Notes'Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled in.
This parameter refers to the batch size setting when training m2tRNA. The size of this batch size is related to your RAM. '  

```'--Learning_rate', type=float, default=0.001```  
Notes'Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled in.
This parameter sets the learning rate for training m2tRNA.'  

```'--epoch_num', type=int, default=1500```  
Notes'Unless the ‘–Use_default_parameter’ is set to ‘False’, this parameter does not need to be filled 
in. This parameter sets the epoch number for training m2tRNA.'  

```'--you_mRNA_data_path',default="default"```  
Notes'Please enter the absolute path of your mRNA readcount data. This data is a comma-separated csv file, which includes ‘gene_id’ 
and sample numbers. If you have any doubts, please refer to the data structure provided on GitHub. I would like to remind you again, 
please make sure that the gene names in this file exist in the training set we provide. If this parameter is set to default, 
it will automatically read the file named ‘test_mRNA_select.csv’ in the m2tRNA directory.'  
