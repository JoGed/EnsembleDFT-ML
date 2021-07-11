# EnsembleDFT-ML
This code provides the ML training & testing methods used in the paper "Machine learning the derivative discontinuity of density-functional theory" (https://arxiv.org/abs/2106.16075)

#### To run the code we propose to do the following steps:
* Read the requirements.  The python version and necessary packages (+versions) which have been used for the training are listed here
* Copy all files into an arbitrary directory where you create the new folders *TrainingsSets*, *TestSetsSpecial* and *Pics*. 
* Download the *Sets.zip* from XXXXXXXXXXXXXXX. 
  It contains the files 
  1. *DictTorch_X.0.05-0.20-0.50-0.8-0.95-1.gz* -> data used for training, validating and testing
  2. *DensData_H2_11_5.csv* -> data used for the prediction of H_2 dissociation curve
  3. *xc_Jump_0.01-0.02-1.csv* -> data containg 0.01 fractional densities used to verify the uniform jump of the xc potential
* Extract *DictTorch_X.0.05-0.20-0.50-0.8-0.95-1.gz* into the *TrainingsSets* folder, and *DensData_H2_11_5.csv* and * *Vxc_Jump_0.01-0.02-1.csv* into the *TestSetsSpecial* folder 
* For training of models via train.py  we propose to use a file containing all arguments - e.g. named *specsTrainExample* - and to type command ```python train.py @specsTrainExample```
* An example of the specs for training is given by the file *specsTrainExample*
* For testing of models via test.py  we propose to use a file containing all arguments as well - e.g. named *specsTestExample* - and to type command ```python test.py @specstestExample```
* An example of the specs for training is given by the file *specsTestExample*
* Type ``-h`` instead of ``@specsTrainExample`` or ``@specsTestExample`` for a detailed description of all necessary and optional arguments being available
