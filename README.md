# 1.Introduction of m2tRNA
m2tRNA is a tool that uses a machine learning model to convert genome-wide mRNA data into tRNA levels. This model includes biological constraints to make accurate predictions.


## 1.1 m2tRNA framework
![fig1-4 0](https://github.com/dydazy/mRNA2tRNA/assets/46813403/81224c04-a128-4380-8ed6-1e9174495a9c)
fig1.Overview of the m2tRNA framework.  
a, Description of data sources, which includes expression data from the TCGA database (a) and tRNA data from DBtRend (g) (Lee, J. O., Lee, M. & Chung, Y. J. DBtRend: A Web-Server of tRNA Expression Profiles from Small RNA Sequencing Data in Humans. Genes (Basel) 12 (2021). https://doi.org:10.3390/genes12101576).  b, Methods for data preprocessing, with the recommended method indicated by a red star.  c, Transformation of input data, where mRNA expression information is weighted according to the codon usage per gene.  d, The weighted data is then input into the first layer of a fully connected neural network. To reduce computational load, a group or multiple groups of genes of interest can be selected for pre-training m2tRNA.  e, The information obtained in (d) is integrated, then constrained using codons of different amino acids.  f, It is then weighted by previously reported tRNA activity to obtain the trained m2tRNA results.  g, The DBtRend database provides tRNA profiles to validate the results from m2tRNA. 

m2tRNA utilizes machine learning to predict tRNA levels from mRNA data, integrating biological insights (Fig.1, more details in METHODS).  To adapt m2tRNA to mRNA data from various sources, ranked mRNA reads were used as input (Fig.1 b).  Considering the greater number of mRNAs than tRNAs, we employed a Multilayer Perceptron (MLP) for effective dimension transformation.  We then weighted the codon preference information onto the mRNA data (Fig.1 c) as the input for the first layer of the MLP (Fig.1 d).  The first layer output matrix was constrained by amino acid-based information and input into the second layer of MLP (Fig.1 e).  The results were weighted using tRNA activity (Fig.1 f), where tRNA activity was defined as a set of weights based on previous research to describe tRNA levels (17).  Finally, we integrated the information of the codons corresponding to each tRNA gene to obtain the predicted tRNA levels. 


## 1.2 Introduction of this Git repository
We have made the m2tRNA available for testing on this Git repository, and it is compatible with JupyterLab/Jupyter Notebook. Additionally, we have included the replication process for all data presented in the manuscript.

### 1.2.1 Here, we have provided explanations for the content of each file (or folder) that we offer.
```mRNA2tRNA/1.Prepare data/Rawdata_tRNA_mRNA``` The training set, which is split as described in fig1.a, is provided in this folder.

```mRNA2tRNA/Prepare data.ipynb``` This file provides the process of splitting the original data and the process of merging the split data, both of which can be replicated in JupyterLab/Jupyter Notebook.
In detail:  Prior to initiating the use of m2tRNA, it is necessary to first prepare the preliminary dataset. This dataset is primarily utilized for training within the m2tRNA neural network. We furnish all the required datasets for training m2tRNA in the section titled 

```1. Prepare Data```. This dataset encompasses approximately 10,000 sample sets. The mRNA data within these sets is sourced from the TCGA database, while the tRNA levels are derived from the tRend database, ensuring a correspondence between the samples (as depicted in Figure 1).
We recommend running the ```Prepare data.ipynb``` in JupyterLab/Jupyter notebook. We provide scripts in this file to prepare the dataset needed for training. After running, you will obtain a large matrix (filename: ```tRNA_log2_mRNA_rank.csv``` or ```tRNA_log2_norm_mRNA_log2_norm.csv```,The file size is approximately ```2,700 MB```.), where rows and columns represent genes and samples respectively. The first 429 rows of the synthesized data frame consist of tRNA expression data, and the subsequent data is mRNA data.

```mRNA2tRNA/2.Model``` 
```mRNA2tRNA/2.Model/Data/``` This file provides the biological information that needs to be imported when running m2tRNA.
In detail:  We provide the m2tRNA for testing and other necessary files for its operation in the ```2.Model``` folder. This includes the main body of m2tRNA in ```2.Model.ipynb```, as well as files containing biological information related to m2tRNA, such as codon usage frequency, tRNA gene activity, and CDS length. We recommend running these directly on JupyterLab/Jupyter Notebook.
```mRNA2tRNA/2.Model/2.Model.ipynb``` This file provides the basic components of m2tRNA, including the import of training data, the processing procedure, the code of m2tRNA, and the training process of m2tRNA.
It’s worth noting that the training set for this test version of m2tRNA is composed by default of ~17,000 genes to infer tRNA levels. If you wish to test your own samples, I suggest you organize the training set and only select those genes that are detected in your samples for training.

```mRNA2tRNA/Z_figure data in paper/``` This folder provides the process for replicating the hands-on content.
In detail: In order to ensure data continuity, we have made every effort to provide the process for reproducing the results mentioned in the manuscript. However, we have not provided some of the larger and less important intermediate process files.





















