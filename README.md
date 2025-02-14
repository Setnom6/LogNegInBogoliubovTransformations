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

# Typical workflow

The file 'exampleOfUse.ipynb' shows a typical workflow of the problem. One starts by defining an object of type 'LogNegManager' which will store the main attributes of the simulation, as the kind of initial states, the matrix transformation in terms of the Bogoliubov coefficients or the number of modes. 

Then one can compute (or load if they were compute earlier) different kind of calculations using the Log Negativity trhough the methods predefined in the LogNegManager class.
=======
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
        "LogNegDifference",
        "OneByOneForAGivenMode",
        "OccupationNumber"
    ],
    "tryToLoad": true,
    "saveFig": true
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
- ```arrayParameters```: (List(float)) List containing the relevant parameters for each type of initial State. For each element of the list, a different initial state is created (all of the same type). Vacuum does not need this parameter, for Thermal the array contains the different temperatures, for the Squeezed states the array contains the squeezing intensities.
- ```dataDirectory```: (string) Path to the directory where the transformation matrices (alphas and betas) are stored.
- ```plotsDirectory```: (string) Path to the directory where the generated plots will be stored.
- ```plotsDataDirectory```: (string) Path to the directory where the data for the plots will be stored
- ```specialModes```: List(int) List of modes used for the simulation 'OneByOneForAGivenMode' (see below) the lowest mode is $1$.
- ```listOfWantedCOmputations```: List(string) List of the required computations. Possible values are:
    - ```FullLogNeg```: For each mode, computes the logarithmic negativity taking that mode as partA and all the others as partB.
    - ```HighestOneByOne```: (Only performed if there is just one IN state) Computes the highest one-to-one logarithmic negativity for each mode and the partner mode that gives the highest value.
    - ```OddVSEven```: Computes the logarithmic negativity for the even modes vs the odd modes and vice versa. That is, for each mode, computes the logarithmic negativity taking that mode as partA, if the mode is even, then partB is all the odd modes, if the mode is odd, then partB is all the even modes. 
    - ```LogNegDifference```: Computes the difference in the logarithmic negativity between the state after the transformation and the state before the transformation.
    - ```OccupationNumber```: Computes the occupation number for each mode of the state.
    - ```OneByOneForAGivenMode```: (Only performed if there is just one IN state) For the modes given in the ```specialModes``` list, computes the one-to-one logarithmic negativity with all the other modes.
- ```tryToLoad```: (bool) If true it looks for the last computations matching the type in ```plotsDataDirectory```.
- ```saveFig```: (bool) If true, in addition of create the data, the simulation creates the figures of each simulation.

# Requirements

The repository uses a copy of ```qgt.py``` taken from https://github.com/Setnom6/Quantum-Gaussian-Information-Toolbox-v2.git (last copy taken on December 2024). Therefore the packages needed are

- Numpy
- Scipy
- Matplotlib

# Warnings

- At the moment, LogNegManager is unable of load data from ```OneToOneForAGivenMode``` so if it is asked to show it it will always compute it.
- The directories ```plotsDirectory``` and ```plotsDataDirectory``` will be created if they do not exist. On the other hand the transformation matrix should be in the specified ```dataDirectory``` with the right format to be extracted.
- The script "ExampleOfUse.ipynb" is outdated. It can be executed but it use an older version of LogNegManager where the plotting and load of data was not incorporated.
