
MSSN
===========================

****

Multimodal Spatio-temporal Stratification Network (MSSN) combines the power of systems biology and deep learning. To the best of our knowledge, our ensembled multimodality framework is the first computational tool that jointly considers AT[N] dynamics, structural network (Net), and clinical assessments for population stratification, integrating interactive and diffusive nature of biomarkers into neural networks.

If you use this code, please cite: https://doi.org/xxxx-xxxx.
 
| Project Name | Authors                                                          |
| ---- |------------------------------------------------------------------|
| ATN_Auto | Enze Xu (xue20@wfu.edu), Jingwen Zhang (zhanj318@wfu.edu) et al.|

| Date       | Version | Version name | Comments                                                    |
|------------|---------|--------------|-------------------------------------------------------------|
| 11/17/2021 | v1.0    | Initial      |                                                             |
| 12/13/2021 | v2.0    | Alpha        | raw 320\*2\*N in data_x                                     |
| 12/15/2021 | v3.0    | Beta         | try longer data_x, but failed                               |
| 12/21/2021 | v4.0    | Gamma        | modeling result is matched and shape of data_x in 320\*9\*N |
| 12/27/2021 | v5.0    | Delta        | modified Gamma                                              |
| 1/4/2021   | v6.0    | Epsilon      | data_y is in shape 320\*9\*1 (the first 1 column)           |
| 1/5/2021   | v7.0    | Zeta         | data_y is in shape 320\*9\*7                                |
| 1/5/2021   | v8.0    | Eta          | data_y is in shape 320\*9\*2 (the first 2 column)           |
| 1/6/2021   | v9.0    | Theta        | data_y is in shape 320\*5\*2                                |
| 1/7/2021   | v10.0   | Iota         | data_y is in shape 320\*1\*2                                |
| 1/10/2021  | v10.5   | Final        | based on eta                                                |

| Python Version | Platform |
| ---- | ---- |
| python3.5 / 3.6 / 3.7 / 3.8 | Linux / Windows / MacOS |

****
# Catalog

* [1 Getting started](#1-getting-started)
* [2 Questions, suggestions and improvements](#2-questions-suggestions-and-improvements)

****

# 1 Getting started

```shell
$ git clone https://github.com/chenm19/MSSN.git
$ cd MSSN
# A virtual environment for python is encouraged
$ pip install -r requirements.txt
$ python run.py -h
# optional arguments:
#   -h, --help              show this help message and exit
#   --num NUM               number of training
#   --comment COMMENT       any comment displayed on record
#   --data DATA             dataset of data_x ("$DATA_TYPE$ID", $DATA_TYPE could be "Alpha", "Gamma", "Delta", "Epsilon", ..., "Iota"; $ID could be 1,2,3,4. e.g. "delta1", "eta2")
#   --kmeans KMEANS         time of doing kmeans as base before training
#   --k K                   number of clusters needed
#   --clear CLEAR           whether to clear the proposed file
#   --main_epoch MAIN_EPOCH iteration in main algorithm
#   --alpha ALPHA           alpha parameter in model (not dataset names)
#   --beta BETA             beta parameter in model (not dataset names)
#   --h_dim H_DIM           h_dim_FC & h_dim_RNN
$ python run.py -data eta2 --num 1 --k 6
# It is normal to see huge warnings at this step, but no need to worry.
# It may cost several minutes.
$ cat record/data=eta2_alpha=0.00001_beta=0.1_h_dim=8_main_epoch=1000/record_eta2.csv
# For each training (number is defined by "--num"), one record will occupy one line.
$ ls saves/data=eta2_alpha=0.00001_beta=0.1_h_dim=8_main_epoch=1000/1/
# The graphs and results created by training.

```

****

# 2 Questions, suggestions and improvements

Email xue20@wfu.edu

