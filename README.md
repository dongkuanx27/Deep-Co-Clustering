# Deep-Co-Clustering

This is a reference implementation for Deep Co-Clustering (SDM'19).

Please feel free to contact Dongkuan Xu (dux19@psu.edu) if you have any question.

# Main function
The main file is /DeepCC/Code/main_coil20.py. It is for the Coil20 dataset.

Other datasets used in our paper are included in /DeepCC/Data.

# Run DeepCC on your datasets
It is easy to run DeepCC on your datasets by changing the data_path in main_coil20.py (Line 13).

Please also change the first dimensions of 'ae_config' and 'ae_col_config' when for other datasets (Line 19, 21).

'ae_config' and 'ae_col_config' are the autoencoder structures for instances and features respectively. 

# Input and Output
The input is the data in the form of a matrix. Rows and columns represent instances and features respectively.

The output are the accuracy and NMI of each iteration.

# Notes
Another important file is: /Code/core/paegmm/kddcup10/kddcup10_pae_gmm.py. This file contains the main body of DeepCC (including the hyperparameter setting, Line 254-257).

The neural network structure and hyperparameter setting of DeepCC for different datasets uesed in our paper are shown in Table-1 and Table-2 of 'SDM2019_DeepCC_Supplemental_Materials.pdf' respectively.

# Citing
Please consider citing the following paper if you find DeepCC useful for your research:

@inproceedings{xu2019de,
  
  title={Deep Co-Clustering},
  
  author={Xu, Dongkuan and Cheng, Wei and Zong, Bo and Ni, Jingchao and Song, Dongjin and Yu, Wenchao and Chen, Yuncong and Chen, Haifeng and Zhang, Xiang},
  
  booktitle={Proceedings of the 2019 SIAM International Conference on Data Mining},
  
  year={2019},
  
  organization={SIAM}
}
