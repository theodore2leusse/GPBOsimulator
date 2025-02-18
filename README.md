This job was done by Théodore DE LEUSSE as part of his internship in the SciNeurotechLab supervised by Marco BONIZZATO.
This internship took place from the September 3rd 2024 to February 17th 2025.

# GPBOsimulator


## Objective    

In this folder, our objective is to perform simulations of neurostimulation with different settings and to evaluate the performance of these settings. This folder allows us to quantify the performance of the chosen settings based on the following criteria:
- We want each iteration of the Bayesian optimization to be short in duration.
- We want the complexity of our surrogate model to be low: a classical GP has cubic complexity with respect to the size of the training dataset, and we want the complexity to be as low as possible.
- We want our optimization to be sample-efficient, converging to the optimum with a minimum number of queries, i.e., in a minimum number of iterations.
    - Does the model find the optimum?
    - How do the exploration and exploitation evolve during the simulation?

The main file is `SimGPBO.py`, where you can find the class `SimGPBO`, which will allow us to run simulations and save all important information into `.npz` files. These files can be found in the `results` folder.

`SimGPBO` class performs simulations on actual data maps that have been extracted from `.mat` files (data available here: https://osf.io/54vhx/) using the `Dataset` class. This class is used in the `simulation.ipynb` notebook.

`Postprocessor` class performs post processing with the `.npz` files in the `results` folder. This class is used in the `results.ipynb` notebook.

`QueriesInfo` class allows to use a SVGP (sparse variational gaussian process) that we will use in the methods `estimated_gpytorch_gpbo` & `estimated_hp_gpytorch_gpbo` from `SimGPBO` class. 

`NNasOnlineGP` class allows to use a NN as an online GP that we will use in the method `NN_gpbo` from `SimGPBO` class.


## Overview

This folder contains:
- 4 python files : `DataSet.py`, `NNasOnlineGP.py`, `NNutils.py`, `Postprocessor.py`, `QueriesInfo.py` & `SimGPBO.py`,
- 3 jupyter notebooks : `simulation.ipynb`, `simuNN.ipynb` et `results.ipynb`.
- 1 folder dataset wich stores lots of different `*.pth` and `*.pkl` datasets.
- 1 folder model that stores weigths of models in `*.pth` files.

The project is organized as follows:

``` bash
/📂GPBOsimulator|
├── 📄DataSet.py            # Contains the dataset class and methods for handling data from the `.mat` files ( data here : https://osf.io/54vhx/ )
├── 📄NNasOnlineGP.py       # class that allow us to use a NN as a surrogate model for bayesian optimisation (cf. `NN_gpbo` method in SimGPBO )
├── 📄NNutils.py            # Utils for using NN for BO
├── 📄Postprocessor.py      # Class used for post process the `.npz files in `results`.
├── 📄QueriesInfo.py        # class that define our SVGP (cf. `estimated_gpytorch_gpbo` & `estimated_hp_gpytorch_gpbo` methods in `SimGPBO` )
├── 📄SimGPBO.py            # class used to set and run the simulations
├── 📄simulation.ipynb      # Jupyter notebook that uses `SimGPBO` class to set & run simulations.
├── 📄results.ipynb         # Jupyter notebook that uses `Postprocessor` class to postprocess the simulations.
├── 📄simuNN.ipynb          # Jupyter notebook that uses `NNasOnlineGP` class to compare the NN surrogate model with a standard GP.
├── 📄README.md             # Project documentation and overview.
├── 📄LICENSE               # License information for the project.
|
├── 📂dataset/              # Stores all datasets
│   ├── 📂emgALLnhp/        # Contains training/validation/test datasets for all EMGs of one non human primate. (Cf. `generate_dataset.ipynb`)
│   ├── 📂emgALLrat/        # Contains training/validation/test datasets for all EMGs of one rat. (Cf. `generate_dataset.ipynb`)
│   ├── 📂emgCUSTOMnhp/     # Contains training/validation/test custom datasets for non-human primates. (Cf. `generate_dataset.ipynb`)
│   ├── 📂pkl_files/        # Pickle files for storing processed data. (Cf. DataGen class)
│   ├── 📂single_map/       # Single map data files. (Cf. DataGen class)
│   ├── 📂test_nhp/         # Test datasets for non-human primates. (Cf. `generate_dataset.ipynb`)
│   ├── 📂test_rat/         # Test datasets for rats. (Cf. `generate_dataset.ipynb`)
|
├── 📂model/                # Model storage
│   ├── 📄...               # models. (Cf. `nn_training_and_results.ipynb`)
|
├── 📂results/              # simulations' results storage
│   ├── 📂old/              # Archive of old models.
│   ├── 📄...               
|
├── 📂gif/                  # Gif storage
│   ├── 📂image/            # Images for the gif 
│   ├── 📄...               # Gif 
```

## Installation

To install NNonlineGP, clone the repository.