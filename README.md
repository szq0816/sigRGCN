# sigRGCN: A robust residual graph convolutional network for scRNA-seq data clustering
# ====== Requirements =====

Running this code involves packages in this file：

requriements.txt


# ======= Run ========

If you want to run this code, enter the command：

cd deeprobust

cd graph

Python  test_prognn.py

# ===== file ======

dataset :   the data set involved in the paper

deeprobust :  the model code in the paper

deeprobust/graph/—_init_.py : Initialization file

deeprobust/graph/Dataset.py : Data reading and data preprocessing

deeprobust/graph/GNN.py ： Residual graph convolutional networks

deeprobust/graph/node_embedding_attack.py  ： Noise injection into cell-graph

deeprobust/graph/preprocessH5.py : Preprocessing of datasets

......


