# Logarithmic Negativity in Bogoliubov Transformations
 Some methods which produces meaningful results from the Bogoliubov coefficients of a transformation using the module QGT_v2 

# Requisites

- The module qgt.py was obtained from https://github.com/Setnom6/Quantum-Gaussian-Information-Toolbox-v2 in its November 2024 version.
- The following modules are also necessary: pylab, numpy, enum, typing, os, datetime, re, matplotlib
- To use the methods following the example the structure of folders and files should be as:

```plaintext

project-root/
|--README.md
|--.gitignore
|--qgt.py
|--LogNegManager.py
|--exampleOfUse.ipynb
|--MatrixTransformationData/
|  |--alpha-n#MODES#-1.txt
|  |-- ...
|  |--alpha-n#MODES#-MODES.txt
|  |--beta-n#MODES#-1.txt
|  |-- ...
|  |--beta-n#MODES#-MODES.txt
|--plots/
|  |--MatrixTransformationDataPlots/
|--plotsData/
|  |--MatrixTransformationDataPlotsData/

```
# Logarithmic Negativity Simulation

This repository provides a simulation framework for computing logarithmic negativity in quantum systems with multiple modes. The simulation is configured using a JSON file and executed using the following command:


```sh
python simulate.py config.json
```

# Configuration file

The simulation is controlled by a configuration file in JSON format. Below is an example:

```json

{
    "MODES": 1024,
    "instant": 10,
    "inStateType": "Thermal",
    "arrayParameters": [12.0],
    "dataDirectory": "./sim-1024-1plt-dL0375-k12-5/",
    "plotsDirectory": "./plots/1024-1plt-plots/",
    "plotsDataDirectory": "./plotsData/1024-1plt-data/",
    "specialModes": [1, 2, 3, 4, 5],
    "listOfWantedComputations": [
        "FullLogNeg",
        "HighestOneByOne",
        "OddVSEven",
        "SameParity",
        "LogNegDifference",
        "OneByOneForAGivenMode",
        "OccupationNumber",
        "JustSomeModes",
    ],
    "tryToLoad": false,
    "saveFig": true,
    "parallelize": true
}
```

## Parameters

- ```MODES```: (int) number of modes in the system.
- ```instant```: (int) Index of the transformation matrix to be used, typically the last one. If number is too high, the transformation used will be the last one.
- ```inStateType```: (string) Initial state type. Possible values are:
  - ```Vacuum```: Vacuum state
  - ```Thermal```: Thermal state (lower states have lower frequencies and therefore more occupation number)
  - ```OneModeSqueezed```: One Mode squeezed state. The squeezing intensity apply equally to each mode.
  - ```TwoModeSqueezed```: Two Mode squeezed state. For each pair of subsequent modes they are squeezed with the same squeezing intensity.
  - ```OneModeSqueezedFixedTemp```: One Mode Squeezed states with an initial temperature. The temperature (same for all initial states) should be given as a new argument in the json file as ```"temperature"```.
  - ```ThermalFixedOneModeSqueezing```: Thermal states with one mode squeezing intensity. The squeezing intensity (same for all initial states) should be given as a new argument in the json file as ```"squeezingIntensity"```.
- ```arrayParameters```: (List(float)) List containing the relevant parameters for each type of initial State. For each element of the list, a different initial state is created (all of the same type). Vacuum does not need this parameter, for Thermal the array contains the different temperatures, for the Squeezed states the array contains the squeezing intensities.
- ```dataDirectory```: (string) Path to the directory where the transformation matrices (alphas and betas) are stored.
- ```plotsDirectory```: (string) Path to the directory where the generated plots will be stored.
- ```plotsDataDirectory```: (string) Path to the directory where the data for the plots will be stored
- ```specialModes```: List(int) List of modes used for the simulation 'OneByOneForAGivenMode' (see below) the lowest mode is $1$.
- ```listOfWantedCOmputations```: List(string) List of the required computations. Possible values are:
    - ```FullLogNeg```: For each mode, computes the logarithmic negativity taking that mode as partA and all the others as partB.
    - ```HighestOneByOne```: (Only performed if there is just one IN state) Computes the highest one-to-one logarithmic negativity for each mode and the partner mode that gives the highest value.
    - ```OddVSEven```: Computes the logarithmic negativity for the even modes vs the odd modes and vice versa. That is, for each mode, computes the logarithmic negativity taking that mode as partA, if the mode is even, then partB is all the odd modes, if the mode is odd, then partB is all the even modes. 
    - ```SameParity```: Computes the logarithmic negativity for the even modes vs the rest of even modes and vice versa. That is, for each mode, computes the logarithmic negativity taking that mode as partA, if the mode is even, then partB is the rest of even modes, if the mode is odd, then partB is the rest of odd modes. 
    - ```LogNegDifference```: Computes the difference in the logarithmic negativity between the state after the transformation and the state before the transformation.
    - ```OccupationNumber```: Computes the occupation number for each mode of the state.
    - ```OneByOneForAGivenMode```: (Only performed if there is just one IN state) For the modes given in the ```specialModes``` list, computes the one-to-one logarithmic negativity with all the other modes.
    - ```JustSomeModes```: Computes the full logarithmic negativity for the modes specified in ```specialModes```
- ```tryToLoad```: (bool) If true it looks for the last computations matching the type in ```plotsDataDirectory```.
- ```saveFig```: (bool) If true, in addition of create the data, the simulation creates the figures of each simulation.
- ```parallelize```: (bool) If true, the computations of the logarithmic negativity will be made in parallel using ```joblib```. If your system does not support simple parallelization (as a cluster with SLURM) set it to ```false´´´.

# Warnings

- At the moment, LogNegManager is unable of load data from ```OneToOneForAGivenMode``` so if it is asked to show it it will always compute it.
- The directories ```plotsDirectory``` and ```plotsDataDirectory``` will be created if they do not exist. On the other hand the transformation matrix should be in the specified ```dataDirectory``` with the right format to be extracted.
- The script "ExampleOfUse.ipynb" is outdated. It can be executed but it use an older version of LogNegManager where the plotting and load of data was not incorporated.
- The use of initial states of type ```OneModeSqueezedFixedTemp``` and ```ThermalFixedOneModeSqueezing``` are still under fixing as change the temperature (or the squeezing intensity) parameter in the json file will load the previous computed files for this type of simulation.
- Use the function ```parallelize``` carefully if your are not sure if joblib with simple setting would work on your hardware.
- In general, try to not load data from precomputed simulations as this functionality needs a general fixing.
