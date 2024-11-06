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
