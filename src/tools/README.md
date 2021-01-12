## Here are puporses of designing these tools

1. modelVisualier.py is designed to help you scale models in read.py quickly. So you won't want to waste your time on running simulation again and again for just adjusting mesh scaling parameters.

2. animComparator.py is designed to help you compare the true PD and PN simulator in one image. So, you can have a reference when you are generating data or you want to compare them to PD-NN's result.

3. kmenasClusterGenerator.py is used to generate a model's cluster file and put that file to the ```ProjectRoot/MeshModels/SavedClusters``` 
folder. The training process needs this cluster file.

4. modelPruner.py is used to prune the generated NN under the ```ProjectRoot/TrainedNN``` folder.

5. pd_standalone.py and pn_standalone.py are used to provide a standard standalone implementation for the simulators, for the
purpose of clarity and debugging.

