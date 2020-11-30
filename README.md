**Non-linear Dynamics -- GNN PDPN**

## how to use network

* pdfix.py is the file that I use to genertae pd data, be careful to set the max iteration to be 50 and the self.m_weight_positional = 10000000.0

* pn2.py is the file that I use to genertae data, you can set the 'animation frame(current: 50)' and animation series(current: 20). There is a boolean variable to control where to put into the output file (Outputs for training data, Outputs_T for testing data).

* train.py is the tarining python file, in this file. You can set the learning rate, the dropout, epoch and other parameters. After training, it will save the model parameters in the set name.

* utils_gcn.py is the util file, reading data is set here.

* layers and GCN_net.py is place to set the network.

* Get_A.py is the file that I use to load the model parameter andf test data and get animation series. It will generate pd pos(2) delta pos(2) add(pd, delta) (2)

* pn_test.py is the file that I use to visualize the effect of network. I use green(pn), yellow(pd) and red(network) to represent 3 different results. 

## Acknowledgement

* [UPenn CG Lab](http://cg.cis.upenn.edu/)

