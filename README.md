# mRNA2tRNA
### A machine learning based method that converts mRNA data into tRNA data.

## Overview of the algorithm framework
### ![fig1-2 0](https://github.com/dydazy/mRNA2tRNA/assets/46813403/89ac26ed-e573-412c-abe5-097b07f087eb)
### Overview of the algorithm framework. (a) Data sources, including expression data from the TCGA database (a) and tRNA data from DBtRend (f). (b) Data preprocessing methods, with the recommended method marked by a red star. (c) Transformation of input data, weighting mRNA expression information according to codon preference data. (d) Inputting data into the first layer of a fully connected neural network. In practical applications, to simplify calculations, a group or multiple groups of genes of interest can be selected for pre-training of the model. (e) Integrating the information obtained in (d), then using amino acid usage information for constraint, and finally weighting with tRNA ontology activity to obtain the results of the trained model.

