import pylab as pl
import numpy as np
import qgt
from enum import Enum
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime
import re

class InitialState(Enum):
    Vacuum = "vacuum"
    Thermal = "thermal"
    OneModeSqueezed = "oneModeSqueezed"
    TwoModeSqueezed = "twoModeSqueezed"

class TypeOfData(Enum):
    FullLogNeg = "fullLogNeg"
    HighestOneByOne = "highestOneByOne"
    OneByOneForAGivenMode = "oneByOneForAGivenMode"
    OddVSEven = "oddVSEven"
    OccupationNumber = "occupationNumber"


class LogNegManager:
    inState: Dict[int, qgt.Gaussian_state]
    outState: Dict[int, qgt.Gaussian_state]
    MODES: int
    kArray: np.ndarray
    instantToPlot: int
    arrayParameters: np.ndarray
    transformationMatrix: np.ndarray
    plottingInfo: Dict[str, Any]

    def __init__(self, dataDirectory: str, initialStateType: InitialState, MODES: int, instantToPlot: int, arrayParameters: np.ndarray = None):
        """
        Constructor for the LogNegManager class

        Parameters:
        dataDirectory: str
            Directory where the data of the transformation matrix is stored (alphas and betas)
        initialStateType: InitialState
            Type of initial state to be used
        MODES: int
            Number of modes of the system
        instantToPlot: int
            Instant of the transformation matrix to be used (usually the last one)
        arrayParameters: np.ndarray
            Array of parameters to be used in the initial state, each parameter will mean a different initial state. It depends on the initialStateType
            - Vacuum: None
            - Thermal: Array of temperatures
            - OneModeSqueezed: Array of one mode squeezing intensities
            - TwoModeSqueezed: Array of two mode squeezing intensities (applied pairwise)
        """
        self.plottingInfo = dict()
        self.MODES = MODES
        self.instantToPlot = instantToPlot
        if initialStateType is not InitialState.Vacuum:
            self.arrayParameters = arrayParameters
        else:
            self.arrayParameters = None
        self.transformationMatrix = self._constructTransformationMatrix(dataDirectory)
        self.inState = self._createInState(initialStateType)
        self.outState = dict()


    def _constructTransformationMatrix(self, directory: str)-> np.ndarray:
        """
        Constructs the transformation matrix from the data stored in the directory.
        The data should be stored in files named as:
        alpha-nMODES-i.txt
        beta-nMODES-i.txt
        where i is the mode index

        It also creates de kArray, which is an array with the mode indexes (typically an array from 1 to MODES)

        Parameters:
        directory: str
            Directory where the data is stored

        Returns:
        np.ndarray
            Transformation matrix constructed from the data using the formula (eq 39 paper):
            Smatrix = ((alpha*_11 -beta*_11 alpha*_12 -beta*_12 ...) , (-beta_11 alpha_11 -beta_12 alpha_12 ...), ...)
        """
        solsalpha = dict()
        solsbeta = dict()

        a=np.arange(1,self.MODES+1)
        dir = directory
        for i in a:
            solsalpha[i]=pl.loadtxt(dir+"alpha-n"+self.MODES+"-"+str(i)+".txt")
            solsbeta[i]=pl.loadtxt(dir+"beta-n"+self.MODES+"-"+str(i)+".txt")      

        time=len(solsbeta[1][:,0])

        #We now save the data in complex arrays
        time=len(solsbeta[1][:,0])
        self.kArray = np.arange(1, self.MODES + 1)
        calphas_array = np.zeros((time, self.MODES, self.MODES),dtype = 'complex_')
        cbetas_array = np.zeros((time, self.MODES, self.MODES),dtype = 'complex_')
        for t in range(0,time):
            for i1 in range(0,self.MODES):
                for i2 in range(1,self.MODES+1):
                    calphas_array[t, i1, i2-1] = solsalpha[i2][t,1+2*i1]+solsalpha[i2][t,2+2*i1]*1j
                    cbetas_array[t, i1, i2-1] = solsbeta[i2][t,1+2*i1]+solsbeta[i2][t,2+2*i1]*1j
        #Label i2 corresponds to in MODES and i1 to out MODES

        #We now save the array at time we are interested in given by the variable "instant"
        instantToPlot = min(self.instantToPlot, time-1)
        calphas_tot_array = np.zeros((self.MODES, self.MODES),dtype = 'complex_')
        cbetas_tot_array = np.zeros((self.MODES, self.MODES),dtype = 'complex_')
        calphas_tot_array = calphas_array[instantToPlot, :, :]
        cbetas_tot_array = cbetas_array[instantToPlot, :, :]

        #For our simulations
        Smatrix = np.zeros((2*self.MODES, 2*self.MODES), dtype=complex)


        #Constructing the Smatrix out of the alpha and beta complex dicts
        #If we write A_out = Smatrix * A_in, see Eq. 39 of our paper, then
        # Smatrix = ((alpha*_11 -beta*_11 alpha*_12 -beta*_12 ...) , (-beta_11 alpha_11 -beta_12 alpha_12 ...), ...)
        #time = 5
        i = 0
        for i1 in range(0,self.MODES):
            j = 0
            for i2 in range(0,self.MODES):
                Smatrix[i, j] = np.conjugate(calphas_tot_array[i1, i2])
                j = j+1
                Smatrix[i, j] = -np.conjugate(cbetas_tot_array[i1, i2])
                j = j+1
            i=i+1
            j = 0
            for i2 in range(0,self.MODES):
                Smatrix[i, j] = -cbetas_tot_array[i1, i2]
                j = j+1
                Smatrix[i, j] = calphas_tot_array[i1, i2]
                j = j+1
            i=i+1

        return Smatrix

    def setTransformationMatrix(self, transformationMatrix: np.ndarray) -> None:
        """
        Sets the transformation matrix to be used in the calculations if different from the one obtained from the data
        
        Parameters:
        transformationMatrix: np.ndarray
            Transformation matrix to be used
        
        Returns:
            None
        """

        self.transformationMatrix = transformationMatrix

    def checkSymplectic(self) -> bool:
        if self.transformationMatrix is None:
            raise Exception("Transformation matrix not initialized")
        else:
            qgt.Is_Sympletic(self.transformationMatrix,1)


    def _createInState(self, initialStateType: InitialState) -> Dict[int, qgt.Gaussian_state]:
        """
        Creates the initial state to be used in the calculations.
        The state is created according to the initialStateType parameter.
        For each element of the arrayParameters, a different state is created.
        Also, the plottingInfo dictionary is updated with the information about the initial state created.

        Parameters:
        initialStateType: InitialState
            Type of initial state to be used

        Returns:
            Dict[int, qgt.Gaussian_state]
            Dictionary with the initial states created.
            The first index is 1, the second is 2, and so on.
            If no arrayParameters is given, the dictionary will have only one element.
        """
        state = dict()
        self.plottingInfo["InStateName"] = initialStateType.value
        if initialStateType == InitialState.Vacuum:
            # Vacuum initial state assumes no arrayParameters
            state[1] = qgt.Gaussian_state("vacuum", self.MODES)
            self.plottingInfo["NumberOfStates"] = 1
            self.plottingInfo["Magnitude"] = [""]
            self.plottingInfo["MagnitudeName"] = ""
            self.plottingInfo["MagnitudeUnits"] = ""
            return state

        elif initialStateType == InitialState.Thermal:
            # Thermal initial state assumes an array of temperatures and creates a thermal state for each temperature
            for index, temp in enumerate(self.arrayParameters):
                n_vector = [1.0/(np.exp(np.pi*self.kArray[i]/temp)-1.0) for i in range(0, self.MODES)] if temp > 0 else [0 for i in range(0, self.MODES)]
                state[index+1] = qgt.elementary_states("thermal", n_vector)

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = " for T "
            self.plottingInfo["MagnitudeUnits"] = "K"
            return state

        elif initialStateType == InitialState.OneModeSqueezed:
            # OneModeSqueezed initial state assumes an array of squeezing intensities and creates an intial state with each mode equally squeezed for each intensity
            for index, intensity in enumerate(self.arrayParameters):
                intensity_array = [intensity for i in range(0, self.MODES)]
                state[index+1] = qgt.elementary_states("squeezed", intensity_array)

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = " for Sqz intensity "
            self.plottingInfo["MagnitudeUnits"] = ""
            return state

        elif initialStateType == InitialState.TwoModeSqueezed:
            # TwoModeSqueezed initial state assumes an array of squeezing intensities and creates an intial state with each pair of consecutive modes squeezed for each intensity
            for index, intensity in enumerate(self.arrayParameters):
                state[index+1] = qgt.Gaussian_state("vacuum", self.MODES)
                for j in range(0, self.MODES, 2):
                    state[index+1].two_mode_squeezing(intensity, 0, [j, j+1])
    
            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = " for Sqz intensity "
            self.plottingInfo["MagnitudeUnits"] = ""
            return state

        else:
            raise ValueError("Unrecognized inStateName")
        

    def performTransformation(self) -> None:
        """
        Performs the transformation of the initial states using the transformation matrix, 
        assuming it is given in termos of the bogoliubov coefficients.
        """
        if self.transformationMatrix is None:
            raise Exception("Transformation matrix not initialized")
        
        if self.inState is None:
            raise Exception("Initial state not initialized")
        
        for index in range(1, self.plottingInfo["NumberOfStates"]+1):
            self.outState[index] = self.inState[index].copy()
            self.outState[index].apply_Bogoliubov_unitary(self.transformationMatrix)

    def computeFullLogNeg(self, inState: bool = False) -> dict[int, np.ndarray]:
        """
        Computes the full logarithmic negativity for the states.
        That is, for each mode, computes the logarithmic negativity taking that mode as partA and all the others as partB.

        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        Returns:
        dict[int, np.ndarray]
            Dictionary with the full logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the full logarithmic negativity for each mode.
            (state i, full log neg of mode j -> fullLogNeg[i][j])
        """
        fullLogNeg = dict()
        stateToApply = self.outState if not inState else self.inState
        for index in range(self.plottingInfo["NumberOfStates"]):
            fullLogNeg[index+1] = np.zeros(self.MODES)


        for i1 in range(0,self.MODES):
            original_list = [i for i in range(0, self.MODES)]
            partA = i1  # Change this to the element you want to remove
            # Create a new list without the specified element
            partB = [x for x in original_list if x != partA]
            partA = [partA]
            for index in range(self.plottingInfo["NumberOfStates"]):
                fullLogNeg[index+1][i1] = stateToApply[index+1].logarithmic_negativity(partA, partB)

        return fullLogNeg


    def computeHighestOneByOne(self, inState: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Just for one IN state, computes the highest one-to-one logarithmic negativity for each mode and the partner mode that gives the highest value.
        
        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        Returns:
        Tuple[np.ndarray, np.ndarray]
            Tuple with two arrays:
            - First array: highest values of the one-to-one logarithmic negativity for each mode
            - Second array: partner mode that gives the highest value for each mode
        
        """
        if self.plottingInfo["NumberOfStates"] > 1:
            raise ValueError("This method is only suitable for one IN state.")
        
        stateToApply = self.outState if not inState else self.inState
        lognegarrayOneByOne: Dict[int, Dict[int,float]] = {}
        maxPartners: List = []
        maxValues: List = []

        for i1 in range(self.MODES):
            lognegarrayOneByOne[i1] = {}
            partA = [i1]
            for i2 in range(self.MODES):
                partB = [i2]
                if partB == partA:
                    lognegarrayOneByOne[i1][i2] = 0.0
                else:
                    lognegarrayOneByOne[i1][i2] = stateToApply[1].logarithmic_negativity(partA, partB)
            maxPartners.append(max(lognegarrayOneByOne[i1], key=lognegarrayOneByOne[i1].get))
            maxValues.append(lognegarrayOneByOne[i1][maxPartners[i1]])     

        return np.array(maxValues), np.array(maxPartners)


    def computeOneByOneForAGivenMode(self, mode, inState: bool = False) -> Dict[int, np.ndarray]: 
        """
        Computes the one-to-one logarithmic negativity for a given mode with all the others.

        Parameters:
        mode: int
            Mode to be used as partA
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        Returns:
        Dict[int, np.ndarray]
            Dictionary with the one-to-one logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the one-to-one logarithmic negativity for the given mode with each other mode.
            (state i, one-to-one log neg between given mode and mode j -> lognegarrayOneByOne[i][j])
        """
        lognegarrayOneByOne: Dict[int, np.ndarray] = dict()
        stateToApply = self.outState if not inState else self.inState
        mode -=1

        for index in range(1, self.plottingInfo["NumberOfStates"]+1):
            lognegarrayOneByOne[index] = np.zeros(self.MODES)
            partA = [mode]
            for i2 in range(self.MODES):
                partB = [i2]
                if partB == partA:
                    lognegarrayOneByOne[index][i2] = 0.0
                else:
                    lognegarrayOneByOne[index][i2] = stateToApply[index].logarithmic_negativity(partA, partB)

        return lognegarrayOneByOne
    
    def computeOddVSEven(self, inState: bool = False) -> Dict[int, np.ndarray]:
        """
        Computes the logarithmic negativity for the even modes vs the odd modes and vice versa.
        That is, for each mode, computes the logarithmic negativity taking that mode as partA,
        if the mode is even, then partB is all the odd modes, if the mode is odd, then partB is all the even modes. 

        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        Returns:
        Dict[int, np.ndarray]
            Dictionary with the logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the logarithmic negativity for each mode.
            (state i, log neg of mode j -> lognegarrayOneByOne[i][j])
        """
        evenFirstModes = np.arange(0, self.MODES-1, 2)
        oddFirstModes = np.arange(1, self.MODES, 2)

        stateToApply = self.outState if not inState else self.inState

        # From the plots one can see that even correlates with even and have no correlation with odds modes and vive versa

        logNegEvenVsOdd = dict()
        for index in range(self.plottingInfo["NumberOfStates"]):
            logNegEvenVsOdd[index+1] = np.zeros((self.MODES,))

        for stateIndex in range(self.plottingInfo["NumberOfStates"]):
            for index in range(self.MODES):
                partA = [index]
                if index in evenFirstModes:
                    partB = [x for x in evenFirstModes if x != partA[0]]
                else:
                    partB = [x for x in oddFirstModes if x != partA[0]]
                logNegEvenVsOdd[stateIndex+1][index] = stateToApply[stateIndex+1].logarithmic_negativity(partA, partB)

        return logNegEvenVsOdd
    

    def computeOccupationNumber(self, inState: bool = False) -> Dict[int, np.ndarray]:
        """
        Computes the occupation number for each mode of the state.
        
        Parameters:
        inState: bool
            If True, the occupation number is computed for the inState, otherwise for the outState
        
        Returns:
        Dict[int, np.ndarray]
            Dictionary with the occupation number for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the occupation number for each mode.
            (state i, occupation number of mode j -> occupationNumber[i][j])
        """
        occupationNumber = dict()
        stateToApply = self.outState if not inState else self.inState

        for index in range(self.plottingInfo["NumberOfStates"]):
            occupationNumber[index+1] = stateToApply[index+1].occupation_number()

        return occupationNumber
    
    def getFigureName(self, plotsRelativeDirectory: str, typeOfData: TypeOfData, date: str = "", beforeTransformation: bool = False) -> str:
        """
        Assuming the plotsRelativeDirectory is defined from the location of the Jupyter Notebook, the standarized figure name is returned

        Parameters:
        plotsRelativeDirectory: str
            Relative directory where the plots are stored
        typeOfData: TypeOfData
            Type of data to be plotted
        date: str
            Date to be added to the figure name
        beforeTransformation: bool
            If True, the figure name will have "Before" before the typeOfData in the name
        
        Returns:
        str
            Figure name with the standarized format
        """
        before = "Before" if beforeTransformation else ""
        figureName = "./{}{}_{}{}_instant_{}_numOfPlots_{}_date_{}.pdf".format(plotsRelativeDirectory, typeOfData.value, self.plottingInfo["InStateName"],before,self.instantToPlot, self.plottingInfo["NumberOfStates"], date)
        return figureName
    
    def getFileName(self, dataRelativeDirectory: str, typeOfData: TypeOfData, date: str = "", beforeTransformation: bool = False) -> List[str]:
        """
        Assuming the plotsRelativeDirectory is defined from the location of the Jupyter Notebook, 
        the standarized files names where the plot data is stored is returned

        Parameters:
        dataRelativeDirectory: str
            Relative directory where the data is stored
        typeOfData: TypeOfData
            Type of data to be plotted
        date: str
            Date to be added to the figure name
        beforeTransformation: bool
            If True, the figure name will have "Before" before the typeOfData in the name
        
        Returns:
        List[str]
            List of files names with the standarized format. Each element of the list corresponds to a different state (number equals number of arrayParameters)
        """
        before = "Before" if beforeTransformation else ""
        filesNames = []
        for index in range(1, self.plottingInfo["NumberOfStates"]+1):
            fileName = "./{}{}_{}{}_instant_{}_numOfPlots_{}_date_{}".format(dataRelativeDirectory, typeOfData.value, self.plottingInfo["InStateName"],before,self.instantToPlot, self.plottingInfo["NumberOfStates"],date)
            filesNames.append(fileName)
        return filesNames
    
    def saveData(self, dataRelativeDirectory: str, data: Dict[int, np.ndarray], typeOfData: TypeOfData, date: str = "", beforeTransformation: bool = False) -> None:
        """
        Method to save the data in the files with the standarized format.
        At the moment it only saves the data if is of type: FullLogNeg, HighestOneByOne, OddVSEven, OccupationNumber

        Parameters:
        dataRelativeDirectory: str
            Relative directory where the data is stored
        data: Dict[int, np.ndarray]
            Dictionary with the data to be saved
        typeOfData: TypeOfData
            Type of data to be saved
        date: str
            Date to be added to the figure name
        beforeTransformation: bool
            If True, the file name will have "Before" before the typeOfData in the name, as it has data from the initial state
        
        Returns:
            None
        """
        filesNamesToSave = self.getFileName(dataRelativeDirectory, typeOfData, date, beforeTransformation)

        for index, fileName in enumerate(filesNamesToSave):

            if typeOfData == TypeOfData.OneByOneForAGivenMode:
                raise NotImplementedError("For {} data one have to save manually".format(typeOfData.value))
            if self.arrayParameters is not None:
                dataToSave = np.zeros(self.MODES+1)
                dataToSave[0] = self.arrayParameters[index] if self.arrayParameters is not None else 0
                dataToSave[1:] = data[index+1]

            else:
                dataToSave = data[index+1]

            np.savetxt("{}_plotNumber_{}.txt".format(fileName, index+1), dataToSave)
            
    
    def loadData(self, dataRelativeDirectory: str, typeOfData: TypeOfData, beforeTransformation: bool = False) -> Dict[int, np.ndarray]:
        """
        Method to load the data from the files with the standarized format.
        At the moment it only loads the data if is of type: FullLogNeg, HighestOneByOne, OddVSEven, OccupationNumber.

        WARNING: At the moment it only loads the data for the last instant of the transformation matrix. It may happen that
        last time a simulation was run with all the same definitions (initialState, instant, number of states, etc) but with different parameters.
        In that case this method will fail and the data will have to be loaded manually.


        Parameters:
        dataRelativeDirectory: str
            Relative directory where the data is stored
        typeOfData: TypeOfData
            Type of data to be loaded
        beforeTransformation: bool
            If True, the data loaded will be from the initial state, as it has data from the initial state

        Returns:
        Dict[int, np.ndarray]
            Dictionary with the data loaded
        """
        data = dict()

        
        filesNames = self.getFileName(dataRelativeDirectory, typeOfData, date="", beforeTransformation=beforeTransformation)   


        allFiles = [f"./{dataRelativeDirectory}{file}" for file in os.listdir(f"./{dataRelativeDirectory}")]


        matchingFiles = [file for file in allFiles if any(re.search(pattern, file) for pattern in filesNames)]
        matchingFiles.sort(reverse=True)

        numberOfStates = self.plottingInfo["NumberOfStates"]
        selectedFiles = matchingFiles[:numberOfStates]
        selectedFiles.sort()

        for index, fileName in enumerate(selectedFiles):
            dataFile = np.loadtxt(fileName)
            if typeOfData == TypeOfData.OneByOneForAGivenMode:
                return dict()
                
            elif self.arrayParameters is not None:
                arrayParameter = dataFile[0]
                if arrayParameter != self.arrayParameters[index]:
                    print("WARNING: Array parameter not found in the array parameters used to generate the data for {}".format(typeOfData))
                    return dict()   
                data[index+1] = dataFile[1:]
            else:
                data[index+1] = dataFile
            
        return data
        

    def checkIfDataExists(self, dataRelativeDirectory: str, typeOfData: TypeOfData, exactFiles: List = [], beforeTransformation: bool = False) -> bool:
        """
        Checks if the data for the given parameters exists in the directory.
        At the moment is not very efficient as it loads all the data and then checks if it is empty.
        """
        
        dataLoaded = self.loadData(dataRelativeDirectory, typeOfData, exactFiles, beforeTransformation=beforeTransformation)
        if len(dataLoaded) == 0:
            return False
        else:  
            return True

