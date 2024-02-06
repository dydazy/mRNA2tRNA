# mRNA2tRNA
### A machine learning based method that converts mRNA data into tRNA data.

## Overview of the algorithm framework
### ![image](https://github.com/dydazy/mRNA2tRNA/assets/46813403/d9903912-ca06-4a0c-a4b4-3d36814796c0)
### Overview of the m2tRNA framework.  
a, Data source description, including expression data from the TCGA database (a) and tRNA data from DBtRend (g).  b, Data preprocessing methods, with the recommended one marked with a red star.  c, Transformation of input data, weighting mRNA expression information according to codon usage per gene.  d, Inputting weighted data into the first layer of a fully connected neural network.  To simplify the computational load, a group or multiple groups of genes of interest can be selected for pre-training m2tRNA.  Integrate the information obtained in (d), then use codons of different amino acids to constrain it (e), and (f) weight it by previously reported tRNA activity to obtain the trained m2tRNA results.  g, The DBtRend database provides tRNA profiles to validate m2tRNA results.  
 


