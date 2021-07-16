# EnsembleDFT-ML
This code provides the ML training & testing methods used in the paper "Machine learning the derivative discontinuity of density-functional theory" (https://arxiv.org/abs/2106.16075)

#### To run the code follow these steps:

* Create an environment (e.g. conda environment) with the python version and necessary packages (+versions) which have been used for the ML training & testing that are listed in the *requirements* file
* Copy all files into an arbitrary directory where you create the new folders *TrainingsSets*, *TestSetsSpecial* and *Pics*. 
* Download the *Sets.zip* from https://zenodo.org/record/5091466#.YOwD-TqxVH4. 
  It contains the files 
  1. *DictTorch_X.0.05-0.20-0.50-0.8-0.95-1.gz* -> data used for training, validating and testing
  2. *DensData_H2_11_5.csv* -> data used for the prediction of H_2 dissociation curve
  3. *xc_Jump_0.01-0.02-1.csv* -> data containg 0.01 fractional densities used to verify the uniform jump of the xc potential
* Extract *DictTorch_X.0.05-0.20-0.50-0.8-0.95-1.gz* into the *TrainingsSets* folder, and *DensData_H2_11_5.csv* and * *Vxc_Jump_0.01-0.02-1.csv* into the *TestSetsSpecial* folder 
* For training of models via train.py use a file containing all arguments/hyperparameters - e.g. named *specsTrainExample* - and to type command ```python train.py @specsTrainExample``` (an example of the file with the same name is already provided)
* This example file trains for one epoch
* The training should produce tensorboard logs and pytorch lightning checkpoint files
* For testing of models via test.py  we propose to use a file containing all arguments as well - e.g. named *specsTestExample* - and to type command ```python test.py @specsTestExample``` (an example of the file with the same name is already provided)
* Type ``-h`` instead of ``@specsTrainExample`` or ``@specsTestExample`` for a detailed description of all necessary and optional arguments being available
* *TestPics.zip* contains the ouput files obtained by passing the arguments (in specsTrainExample & specsTestExample) for training and testing respectively and should look somewhat similar
* the files m101-m107 in *Models.zip* containing different hyperparameters correspond to the models presented in the paper
