**Non-linear Dynamics -- GNN PDPN**

## how to use network

* Environment: Ubuntu 18/20

* Outs change to (PD_dis - PN_dis)

* This project depends on two pre-compiled platform dependent files: src/Utils/a.so and src/Utils/JGSL_WATER.xxx.so. Please remember to replace them with your own version if current ones do not work.

* The running process: run.py (In training data generating mode) -> train.py -> run.py (In testing data generating mode) -> Get_A.py -> test_visualizer.py.

* run.py is used to generate PD/PD data for the purpose of training or testing.

* train.py is the tarining python file, in this file, you can set the learning rate, the dropout, epoch and other parameters. After training, it will save the model parameters in the set name and save it to TrainedNN folder.

* utils_gcn.py is the util file, reading data is set here.

* Get_A.py is the file used to test and generate testing results like corrected PD displacement delta.

* test_visualizer.py is the file used to visualize the data generated previously by other applications. It now ,in mode 3, can visualize data produced by Get_A.py and show PD, PN and PD-NN results. 

## Acknowledgement

* [UPenn CG Lab](http://cg.cis.upenn.edu/)

