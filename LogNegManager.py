import os
import re
from datetime import datetime
from enum import Enum
from itertools import combinations
from math import comb
from typing import Dict, Any, List, Tuple, Optional

import matplotlib as mpl
import numpy as np
import pylab as pl
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm

import qgt
from partnerMethods import extractHawkingPartner, get_symplectic_from_covariance


class InitialState(Enum):
    Vacuum = "vacuum"
    Thermal = "thermal"
    OneModeSqueezed = "oneModeSqueezed"
    TwoModeSqueezed = "twoModeSqueezed"
    OneModeSqueezedFixedTemp = "oneModeSqueezedFixedTemp"
    ThermalFixedOneModeSqueezing = "thermalFixedOneModeSqueezing"
    TwoModeSqueezedFixedTemp = "twoModeSqueezedFixedTemp"


class TypeOfData(Enum):
    FullLogNeg = "fullLogNeg"
    FullLogNegBefore = "fullLogNegBefore"
    HighestOneByOne = "highestOneByOne"
    OneByOneForAGivenMode = "oneByOneForAGivenMode"
    OneVSTwoForAGivenMode = "oneVSTwoForAGivenMode"
    OddVSEven = "oddVSEven"
    SameParity = "sameParity"
    OccupationNumber = "occupationNumber"
    LogNegDifference = "logNegDifference"
    JustSomeModes = "justSomeModes"
    HawkingPartner = "hawkingPartner"


class LogNegManager:
    inState: Dict[int, qgt.Gaussian_state]
    outState: Dict[int, qgt.Gaussian_state]
    MODES: int
    kArray: np.ndarray
    instantToPlot: int
    arrayParameters: np.ndarray
    transformationMatrix: np.ndarray
    plottingInfo: Dict[str, Any]

    def __init__(self, dataDirectory: str, initialStateType: InitialState, MODES: int, instantToPlot: int,
                 arrayParameters: np.ndarray = None, temperature: Optional[float] = None,
                 squeezingIntensity: Optional[float] = None, parallelize: bool = False):
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
            - OneModeSqueezedFixedTemp: Array of one mode squeezing intensities in a thermal bath with fixed T
            - TwoModeSqueezedFixedTemp: Array of two mode squeezing intensities in a thermal bath with fixed T
            - ThermalFixedOneModeSqueezing: Array of temperatures for initial states being one mode squeezing with a fixed squeezing intensity
        """
        self._temperature = None
        self._squeezing = None
        if temperature is not None:
            self.setTemperature(temperature)
        if squeezingIntensity is not None:
            self.setSqueezing(squeezingIntensity)
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
        self.parallelize = parallelize

    def _execute(self, tasks):
        if self.parallelize:
            return Parallel(n_jobs=5)(delayed(t)() for t in tasks)
        else:
            return [t() for t in tasks]

    def _constructTransformationMatrix(self, directory: str) -> np.ndarray:
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

        a = np.arange(1, self.MODES + 1)
        dir = directory
        for i in a:
            solsalpha[i] = pl.loadtxt(dir + "alpha-n" + str(self.MODES) + "-" + str(i) + ".txt")
            solsbeta[i] = pl.loadtxt(dir + "beta-n" + str(self.MODES) + "-" + str(i) + ".txt")

        time = len(solsbeta[1][:, 0])

        # We now save the data in complex arrays
        time = len(solsbeta[1][:, 0])
        self.kArray = np.arange(1, self.MODES + 1)
        calphas_array = np.zeros((time, self.MODES, self.MODES), dtype=np.complex128)
        cbetas_array = np.zeros((time, self.MODES, self.MODES), dtype=np.complex128)
        for t in range(0, time):
            for i1 in range(0, self.MODES):
                for i2 in range(1, self.MODES + 1):
                    calphas_array[t, i1, i2 - 1] = solsalpha[i2][t, 1 + 2 * i1] + solsalpha[i2][t, 2 + 2 * i1] * 1j
                    cbetas_array[t, i1, i2 - 1] = solsbeta[i2][t, 1 + 2 * i1] + solsbeta[i2][t, 2 + 2 * i1] * 1j
        # Label i2 corresponds to in MODES and i1 to out MODES

        # We now save the array at time we are interested in given by the variable "instant"
        self.instantToPlot = min(self.instantToPlot, time - 1)
        calphas_tot_array = np.zeros((self.MODES, self.MODES), dtype=np.complex128)
        cbetas_tot_array = np.zeros((self.MODES, self.MODES), dtype=np.complex128)
        calphas_tot_array = calphas_array[self.instantToPlot, :, :]
        cbetas_tot_array = cbetas_array[self.instantToPlot, :, :]

        # For our simulations
        Smatrix = np.zeros((2 * self.MODES, 2 * self.MODES), dtype=np.complex128)

        # Constructing the Smatrix out of the alpha and beta complex dicts
        # If we write A_out = Smatrix * A_in, see Eq. 39 of our paper, then
        # Smatrix = ((alpha*_11 -beta*_11 alpha*_12 -beta*_12 ...) , (-beta_11 alpha_11 -beta_12 alpha_12 ...), ...)
        # time = 5
        i = 0
        for i1 in range(0, self.MODES):
            j = 0
            for i2 in range(0, self.MODES):
                Smatrix[i, j] = np.conjugate(calphas_tot_array[i1, i2])
                j = j + 1
                Smatrix[i, j] = -np.conjugate(cbetas_tot_array[i1, i2])
                j = j + 1
            i = i + 1
            j = 0
            for i2 in range(0, self.MODES):
                Smatrix[i, j] = -cbetas_tot_array[i1, i2]
                j = j + 1
                Smatrix[i, j] = calphas_tot_array[i1, i2]
                j = j + 1
            i = i + 1

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

    def invertTransformationMatrix(self) -> None:
        """
        Given that A_in = S @ A_out, this computes the inverse Bogoliubov transformation matrix
        using the relation:

            S^{-1} = Omega^{-1} @ S^† @ Omega

        where S^† is the conjugate transpose of S.
        """

        omegaMatrix = self.inState[1].Omega  # Assumes shape (2N, 2N)
        S = self.transformationMatrix  # The original matrix: A_in = S @ A_out
        S_dagger = S.conj().T  # Conjugate transpose: S^†

        self.transformationMatrix = np.linalg.inv(omegaMatrix) @ S_dagger @ omegaMatrix

    def checkSymplectic(self) -> bool:
        if self.transformationMatrix is None:
            raise Exception("Transformation matrix not initialized")
        else:
            return qgt.Is_Sympletic(self.transformationMatrix, 1)

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
            # Thermal initial state assumes an array of temperatures (in Kelvin )and creates a thermal state for each temperature
            for index, temp in enumerate(self.arrayParameters):
                temperature = 0.694554 * temp  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
                n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                            range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)]
                state[index + 1] = qgt.elementary_states("thermal", n_vector)

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = " for T "
            self.plottingInfo["MagnitudeUnits"] = "K"
            return state

        elif initialStateType == InitialState.OneModeSqueezed:
            # OneModeSqueezed initial state assumes an array of squeezing intensities and creates an intial state with each mode equally squeezed for each intensity
            for index, intensity in enumerate(self.arrayParameters):
                intensity_array = [intensity for i in range(0, self.MODES)]
                state[index + 1] = qgt.elementary_states("squeezed", intensity_array)

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = " for Sqz intensity "
            self.plottingInfo["MagnitudeUnits"] = ""
            return state

        elif initialStateType == InitialState.TwoModeSqueezed:
            # TwoModeSqueezed initial state assumes an array of squeezing intensities and creates an intial state with each pair of consecutive modes squeezed for each intensity
            for index, intensity in enumerate(self.arrayParameters):
                state[index + 1] = qgt.Gaussian_state("vacuum", self.MODES)
                for j in range(0, self.MODES, 2):
                    state[index + 1].two_mode_squeezing(intensity, 0, [j, j + 1])

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = " for Sqz intensity "
            self.plottingInfo["MagnitudeUnits"] = ""
            return state

        elif initialStateType == InitialState.OneModeSqueezedFixedTemp:
            for index, intensity in enumerate(self.arrayParameters):
                intensity_array = [intensity for i in range(0, self.MODES)]
                state[index + 1] = qgt.elementary_states("squeezed", intensity_array)
                assert self._temperature is not None, "A 'temperature' must be defined to use OneModeSqueezedFixedTemp"
                temp = 0.694554 * self._temperature  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
                n_vector = np.array([1.0 / (np.exp(np.pi * self.kArray[i] / temp) - 1.0) for i in
                                     range(0, self.MODES)] if temp > 0 else [0 for i in range(0, self.MODES)])

                state[index + 1].add_thermal_noise(n_vector)

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = f" for Sqz intensity "
            self.plottingInfo["MagnitudeUnits"] = ""
            self.plottingInfo["title"] = f"T = {self._temperature} K"
            return state

        elif initialStateType == InitialState.TwoModeSqueezedFixedTemp:
            for index, intensity in enumerate(self.arrayParameters):
                state[index + 1] = qgt.Gaussian_state("vacuum", self.MODES)
                for j in range(0, self.MODES, 2):
                    state[index + 1].two_mode_squeezing(intensity, 0, [j, j + 1])

                assert self._temperature is not None, "A 'temperature' must be defined to use TwoModeSqueezedFixedTemp"
                temp = 0.694554 * self._temperature  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
                n_vector = np.array([1.0 / (np.exp(np.pi * self.kArray[i] / temp) - 1.0) for i in
                                     range(0, self.MODES)] if temp > 0 else [0 for i in range(0, self.MODES)])

                state[index + 1].add_thermal_noise(n_vector)

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = " for Sqz intensity "
            self.plottingInfo["MagnitudeUnits"] = ""
            return state

        elif initialStateType == InitialState.ThermalFixedOneModeSqueezing:
            for index, temp in enumerate(self.arrayParameters):
                temperature = 0.694554 * temp  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
                n_vector = np.array([1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                                     range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)])
                assert self._squeezing is not None, "A 'squeezingIntensity' must be defined to use ThermalFixedOneModeSqueezing"
                r_vector = [self._squeezing for i in range(self.MODES)]
                state[index + 1] = qgt.elementary_states("squeezed", r_vector)
                state[index + 1].add_thermal_noise(n_vector)

            self.plottingInfo["NumberOfStates"] = len(self.arrayParameters)
            self.plottingInfo["Magnitude"] = self.arrayParameters
            self.plottingInfo["MagnitudeName"] = f" for T "
            self.plottingInfo["MagnitudeUnits"] = "K"
            self.plottingInfo["title"] = f"r = {self._squeezing}"
            return state

        else:
            raise ValueError("Unrecognized inStateName")

    def setTemperature(self, temp: float) -> None:
        self._temperature = temp

    def setSqueezing(self, r: float) -> None:
        self._squeezing = r

    def performTransformation(self) -> None:
        """
        Performs the transformation of the initial states using the transformation matrix, 
        assuming it is given in termos of the bogoliubov coefficients.
        """
        if self.transformationMatrix is None:
            raise Exception("Transformation matrix not initialized")

        if self.inState is None:
            raise Exception("Initial state not initialized")

        for index in range(1, self.plottingInfo["NumberOfStates"] + 1):
            self.outState[index] = self.inState[index].copy()
            self.outState[index].apply_Bogoliubov_unitary(self.transformationMatrix)

    def checkPurity(self, inState: bool = False) -> None:
        stateToCheck = self.inState.copy() if inState else self.outState.copy()
        if stateToCheck is None:
            raise Exception("No state to check")

        for index in range(1, self.plottingInfo["NumberOfStates"] + 1):
            print(f"Purity of state {index}: {stateToCheck[index].purity()}")

    def computeFullLogNeg(self, inState: bool = False, specialModes: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Computes the full logarithmic negativity for the states.
        That is, for each mode, computes the logarithmic negativity taking that mode as partA and all the others as partB.

        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        specialModes: list of int
            Particular modes to consider (incompatible with numberOfModes)

        Returns:
        dict[int, np.ndarray]
            Dictionary with the full logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the full logarithmic negativity for each mode.
            (state i, full log neg of mode j -> fullLogNeg[i][j])
        """
        stateToApply = self.outState if not inState else self.inState
        if specialModes is None:
            numberOfModes = self.MODES
            modesToConsider = [idx for idx in range(numberOfModes)]
        else:
            numberOfModes = len(specialModes)
            modesToConsider = specialModes
            print(modesToConsider)

        fullLogNeg = {i + 1: np.zeros(numberOfModes) for i in range(self.plottingInfo["NumberOfStates"])}

        def task(index, i1):
            return lambda: (
                index + 1,
                i1,
                stateToApply[index + 1].logarithmic_negativity([i1], [x for x in range(self.MODES) if x != i1])
            )

        tasks = [task(index, i1)
                 for i1 in modesToConsider
                 for index in range(self.plottingInfo["NumberOfStates"])]

        results = self._execute(tasks)

        for idx, i1, value in results:
            i1Index = modesToConsider.index(i1)
            fullLogNeg[idx][i1Index] = value

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

        state = self.outState if not inState else self.inState
        modeCount = self.MODES

        def task(i1):
            def inner():
                values = {}
                for i2 in range(modeCount):
                    if i1 == i2:
                        values[i2] = 0.0
                    else:
                        values[i2] = state[1].logarithmic_negativity([i1], [i2])
                maxPartner = max(values, key=values.get)
                return values[maxPartner], maxPartner

            return inner

        tasks = [task(i1) for i1 in range(modeCount)]
        results = self._execute(tasks)
        maxValues, maxPartners = zip(*results)
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
        lognegarrayOneByOne: Dict[int, np.ndarray] = {i: np.zeros(self.MODES) for i in
                                                      range(1, self.plottingInfo["NumberOfStates"] + 1)}
        stateToApply = self.outState if not inState else self.inState
        mode -= 1  # ajustar a índice 0

        def task(index, i2):
            return lambda: (
                index,
                i2,
                0.0 if mode == i2 else stateToApply[index].logarithmic_negativity([mode], [i2])
            )

        tasks = [task(index, i2)
                 for index in range(1, self.plottingInfo["NumberOfStates"] + 1)
                 for i2 in range(self.MODES)]

        results = self._execute(tasks)

        for index, i2, value in results:
            lognegarrayOneByOne[index][i2] = value

        return lognegarrayOneByOne

    def computeOneVSTwoForAGivenMode(self, mode: int, inState: bool = False) -> Dict[int, np.ndarray]:
        """
        Computes the logarithmic negativity between a given mode and all unordered pairs of other modes.

        Parameters:
        mode: int
            Mode to be used as partA.
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState.

        Returns:
        Dict[int, np.ndarray]
            Dictionary where each key corresponds to a state index.
            Each value is an array with shape (n_pairs,), containing the logarithmic negativities between `mode` and each pair (i,j) with i < j ≠ mode.
        """

        mode -= 1  # Ajustar a índice 0
        stateToApply = self.inState if inState else self.outState

        # Todas las combinaciones (i, j) con i < j ≠ mode
        allPairs = [(i, j) for i, j in combinations(range(self.MODES), 2)
                    if mode not in (i, j)]

        nPairs = len(allPairs)
        lognegDict = {i: np.zeros(nPairs) for i in range(1, self.plottingInfo["NumberOfStates"] + 1)}

        def task(stateIndex, pairIndex, i, j):
            return lambda: (
                stateIndex,
                pairIndex,
                stateToApply[stateIndex].logarithmic_negativity([mode], [i, j])
            )

        tasks = [task(stateIndex, pairIndex, i, j)
                 for stateIndex in range(1, self.plottingInfo["NumberOfStates"] + 1)
                 for pairIndex, (i, j) in enumerate(allPairs)]

        results = self._execute(tasks)

        for stateIndex, pairIndex, value in results:
            lognegDict[stateIndex][pairIndex] = value

        return lognegDict

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
        evenFirstModes = np.arange(0, self.MODES - 1, 2)
        oddFirstModes = np.arange(1, self.MODES, 2)
        stateToApply = self.outState if not inState else self.inState
        logNegEvenVsOdd = {i + 1: np.zeros(self.MODES) for i in range(self.plottingInfo["NumberOfStates"])}

        def task(stateIndex, mode):
            def inner():
                partA = [mode]
                if mode in evenFirstModes:
                    partB = [x for x in oddFirstModes if x != mode]
                else:
                    partB = [x for x in evenFirstModes if x != mode]
                value = stateToApply[stateIndex + 1].logarithmic_negativity(partA, partB)
                return stateIndex + 1, mode, value

            return inner

        tasks = [task(stateIndex, mode)
                 for stateIndex in range(self.plottingInfo["NumberOfStates"])
                 for mode in range(self.MODES)]

        results = self._execute(tasks)

        for stateIndex, mode, value in results:
            logNegEvenVsOdd[stateIndex][mode] = value

        return logNegEvenVsOdd

    def computeSameParity(self, inState: bool = False) -> Dict[int, np.ndarray]:
        """
                Computes the logarithmic negativity for the even modes vs the rest of even modes. The same for the odd modes.
                That is, for each mode, computes the logarithmic negativity taking that mode as partA,
                if the mode is even, then partB is the rest of the even modes, if the mode is odd, then partB is the rest of the odd modes.

                Parameters:
                inState: bool
                    If True, the logarithmic negativity is computed for the inState, otherwise for the outState

                Returns:
                Dict[int, np.ndarray]
                    Dictionary with the logarithmic negativity for each state. (indexes 1, 2, ...)
                    Each element of the dictionary is an array with the logarithmic negativity for each mode.
                    (state i, log neg of mode j -> lognegarrayOneByOne[i][j])
                """
        evenFirstModes = np.arange(0, self.MODES - 1, 2)
        oddFirstModes = np.arange(1, self.MODES, 2)
        stateToApply = self.outState if not inState else self.inState
        logNegSameParity = {i + 1: np.zeros(self.MODES) for i in range(self.plottingInfo["NumberOfStates"])}

        def task(stateIndex, mode):
            def inner():
                partA = [mode]
                if mode in evenFirstModes:
                    partB = [x for x in evenFirstModes if x != mode]
                else:
                    partB = [x for x in oddFirstModes if x != mode]
                value = stateToApply[stateIndex + 1].logarithmic_negativity(partA, partB)
                return stateIndex + 1, mode, value

            return inner

        tasks = [task(stateIndex, mode)
                 for stateIndex in range(self.plottingInfo["NumberOfStates"])
                 for mode in range(self.MODES)]

        results = self._execute(tasks)

        for stateIndex, mode, value in results:
            logNegSameParity[stateIndex][mode] = value

        return logNegSameParity

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

        for index in range(1, self.plottingInfo["NumberOfStates"] + 1):
            occupationNumber[index] = stateToApply[index].occupation_number().flatten()

        return occupationNumber

    def computeLogNegDifference(self, logNegArray):
        """
        Computes the difference in the logarithmic negativity between the state after the transformation and the state before the transformation.

        Parameters:
        logNegArray: Dict[int, np.ndarray]
            Dictionary with the logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the logarithmic negativity for each mode.
            (state i, log neg of mode j -> logNegArray[i][j])
        
        Returns:
        Dict[int, np.ndarray]
            Dictionary with the difference in the logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the difference in the logarithmic negativity for each mode.
            (state i, difference in log neg of mode j -> differenceArray[i][j])
        """
        if logNegArray is None:
            logNegArray = self.computeFullLogNeg()
        logNegArrayBefore = self.computeFullLogNeg(inState=True)
        differenceArray = dict()
        for index in range(1, self.plottingInfo["NumberOfStates"] + 1):
            differenceArray[index] = np.zeros(self.MODES)
            for mode in range(self.MODES):
                differenceArray[index][mode] = logNegArray[index][mode] - logNegArrayBefore[index][mode]

        return differenceArray

    def computeHawkingPartnerLogNeg(self, inState: bool = False, specialModes: List[int] = None) -> Dict[
        int, np.ndarray]:
        """
        Computes the full logarithmic negativity between the hawking mode and its partner (each HP basis is independent from the others).
        That is, for each mode, computes the logarithmic negativity taking that Hawking mode as partA and the partner as partB, tracing out the other modes.

        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        specialModes: list of int
            Particular modes to consider

        Returns:
        dict[int, np.ndarray]
            Dictionary with the Hawking Partner logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the HP logarithmic negativity for each mode.
            (state i, HP log neg of mode j -> fullLogNeg[i][j]) Only works for 1 state i
        """

        if self.plottingInfo["NumberOfStates"] != 1:
            raise ValueError("Hawking-partner calculation only works for 1 simulation at once")

        if self.plottingInfo["InStateName"] in [InitialState.Thermal.value, InitialState.OneModeSqueezedFixedTemp.value,
                                                InitialState.ThermalFixedOneModeSqueezing.value,
                                                InitialState.TwoModeSqueezedFixedTemp.value]:
            raise AttributeError("Hawking-partner calculation only works for pure states")

        results = self.obtainHPLogNegForListOfModes(specialModes, inState)

        HPLogNeg = {i + 1: np.zeros(len(specialModes)) for i in range(self.plottingInfo["NumberOfStates"])}

        for mode, logNeg in results:
            HPLogNeg[1][specialModes.index(mode)] = logNeg

        return HPLogNeg

    def getFigureName(self, plotsRelativeDirectory: str, typeOfData: TypeOfData, date: str = "",
                      beforeTransformation: bool = False) -> str:
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
        figureName = "./{}{}_{}{}_instant_{}_numOfPlots_{}_date_{}.pdf".format(plotsRelativeDirectory, typeOfData.value,
                                                                               self.plottingInfo["InStateName"], before,
                                                                               self.instantToPlot,
                                                                               self.plottingInfo["NumberOfStates"],
                                                                               date)
        return figureName

    def getFileName(self, dataRelativeDirectory: str, typeOfData: TypeOfData, date: str = "",
                    beforeTransformation: bool = False) -> List[str]:
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
        for index in range(1, self.plottingInfo["NumberOfStates"] + 1):
            fileName = "./{}{}_{}{}_instant_{}_numOfPlots_{}_date_{}".format(dataRelativeDirectory, typeOfData.value,
                                                                             self.plottingInfo["InStateName"], before,
                                                                             self.instantToPlot,
                                                                             self.plottingInfo["NumberOfStates"], date)
            filesNames.append(fileName)
        return filesNames

    def saveData(self, dataRelativeDirectory: str, data: Dict[int, np.ndarray], typeOfData: TypeOfData, date: str = "",
                 beforeTransformation: bool = False) -> None:
        """
        Method to save the data in the files with the standarized format.
        At the moment it only saves the data if is of type: FullLogNeg, HighestOneByOne, OddVSEven, SameParity, OccupationNumber, Difference

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

            if typeOfData == TypeOfData.OneByOneForAGivenMode or typeOfData == TypeOfData.OneVSTwoForAGivenMode:
                raise NotImplementedError("For {} data one have to save manually".format(typeOfData.value))

            if typeOfData == TypeOfData.HighestOneByOne:
                dataToSave = np.zeros((2, self.MODES))
                dataToSave[0, :] = data[index + 1][0]
                dataToSave[1, :] = data[index + 1][1]
            else:
                if self.arrayParameters is not None:
                    dataToSave = np.zeros(self.MODES + 1)
                    dataToSave[0] = self.arrayParameters[index] if self.arrayParameters is not None else 0
                    dataToSave[1:] = data[index + 1]
                else:
                    dataToSave = data[index + 1]

            np.savetxt("{}_plotNumber_{}.txt".format(fileName, index + 1), dataToSave)

    def loadData(self, dataRelativeDirectory: str, typeOfData: TypeOfData, beforeTransformation: bool = False) -> Dict[
        int, np.ndarray]:
        """
        Method to load the data from the files with the standarized format.
        At the moment it only loads the data if is of type: FullLogNeg, HighestOneByOne, OddVSEven,SameParity, OccupationNumber, Difference.

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

        filesNames = self.getFileName(dataRelativeDirectory, typeOfData, date="",
                                      beforeTransformation=beforeTransformation)

        allFiles = [f"./{dataRelativeDirectory}{file}" for file in os.listdir(f"./{dataRelativeDirectory}")]

        matchingFiles = [file for file in allFiles if any(re.search(pattern, file) for pattern in filesNames)]
        matchingFiles.sort(reverse=True)

        numberOfStates = self.plottingInfo["NumberOfStates"]
        selectedFiles = matchingFiles[:numberOfStates]
        selectedFiles.sort()

        for index, fileName in enumerate(selectedFiles):
            dataFile = np.loadtxt(fileName)
            if typeOfData == TypeOfData.OneByOneForAGivenMode or typeOfData == TypeOfData.OneVSTwoForAGivenMode:
                return dict()

            elif typeOfData == TypeOfData.HighestOneByOne:
                data[index + 1] = np.zeros((2, self.MODES))
                data[index + 1][0] = dataFile[0, :]
                data[index + 1][1] = dataFile[1, :]
            else:
                if self.arrayParameters is not None:
                    arrayParameter = dataFile[0]
                    if arrayParameter != self.arrayParameters[index]:
                        print(
                            "WARNING: Array parameter not found in the array parameters used to generate the data for {}".format(
                                typeOfData))
                        return dict()
                    data[index + 1] = dataFile[1:]
                else:
                    data[index + 1] = dataFile

        return data

    def checkIfDataExists(self, dataRelativeDirectory: str, typeOfData: TypeOfData,
                          beforeTransformation: bool = False) -> bool:
        """
        Checks if the data for the given parameters exists in the directory.
        At the moment is not very efficient as it loads all the data and then checks if it is empty.
        """

        dataLoaded = self.loadData(dataRelativeDirectory, typeOfData, beforeTransformation=beforeTransformation)
        return len(dataLoaded) > 0

    def performComputations(self, listOfWantedComputations: List[TypeOfData], plotsDataDirectory: str,
                            tryToLoad: bool = True, specialModes: List[int] = []):
        results = {
            "logNegArray": None,
            "logNegArrayBefore": None,
            "highestOneToOneValue": None,
            "highestOneToOnePartner": None,
            "occupationNumber": None,
            "logNegEvenVsOdd": None,
            "logNegSameParity": None,
            "oneToOneGivenModes": None,
            "oneVSTwoForAGivenModes": None,
            "logNegDifference": None,
            "justSomeModes": None,
            "hawkingPartner": None,
        }

        if (self.plottingInfo["InStateName"] == InitialState.OneModeSqueezedFixedTemp.value
                or self.plottingInfo["InStateName"] == InitialState.ThermalFixedOneModeSqueezing.value
                or self.plottingInfo["InStateName"] == InitialState.TwoModeSqueezedFixedTemp.value):
            tryToLoad = False

        for computation in listOfWantedComputations:
            loadData = tryToLoad and self.checkIfDataExists(plotsDataDirectory, computation)

            if loadData:
                print("Loading data for: ", computation.value)

                if computation == TypeOfData.FullLogNeg:
                    results["logNegArray"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.HighestOneByOne:
                    oneToOneData = self.loadData(plotsDataDirectory, computation)
                    results["highestOneToOneValue"] = oneToOneData[1][0]
                    results["highestOneToOnePartner"] = oneToOneData[1][1]

                elif computation == TypeOfData.OccupationNumber:
                    results["occupationNumber"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.OddVSEven:
                    results["logNegEvenVsOdd"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.SameParity:
                    results["logNegSameParity"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.LogNegDifference:
                    results["logNegDifference"] = self.loadData(plotsDataDirectory, computation)
                    results["logNegArray"] = self.loadData(plotsDataDirectory, TypeOfData.FullLogNeg)
                    results["logNegArrayBefore"] = self.loadData(plotsDataDirectory, TypeOfData.FullLogNegBefore)

            else:
                date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                print("Computing data for: ", computation.value)

                if computation == TypeOfData.FullLogNeg:
                    results["logNegArray"] = self.computeFullLogNeg()
                    self.saveData(plotsDataDirectory, results["logNegArray"], computation, date)

                elif computation == TypeOfData.HighestOneByOne:
                    if self.plottingInfo["NumberOfStates"] == 1:
                        results["highestOneToOneValue"], results[
                            "highestOneToOnePartner"] = self.computeHighestOneByOne()
                        oneToOneDict = {
                            1: np.array([results["highestOneToOneValue"], results["highestOneToOnePartner"]])}
                        self.saveData(plotsDataDirectory, oneToOneDict, computation, date)
                    else:
                        print("Highest one by one not computed for this initial state (more than one initial state)")

                elif computation == TypeOfData.OccupationNumber:
                    results["occupationNumber"] = self.computeOccupationNumber()
                    self.saveData(plotsDataDirectory, results["occupationNumber"], TypeOfData.OccupationNumber, date)

                elif computation == TypeOfData.OddVSEven:
                    results["logNegEvenVsOdd"] = self.computeOddVSEven()
                    self.saveData(plotsDataDirectory, results["logNegEvenVsOdd"], TypeOfData.OddVSEven, date)

                elif computation == TypeOfData.SameParity:
                    results["logNegSameParity"] = self.computeSameParity()
                    self.saveData(plotsDataDirectory, results["logNegSameParity"], TypeOfData.SameParity, date)

                elif computation == TypeOfData.LogNegDifference:
                    results["logNegDifference"] = self.computeLogNegDifference(results["logNegArray"])
                    self.saveData(plotsDataDirectory, results["logNegDifference"], TypeOfData.LogNegDifference, date)
                    results["logNegArray"] = self.computeFullLogNeg()
                    self.saveData(plotsDataDirectory, results["logNegArray"], TypeOfData.FullLogNeg, date)
                    results["logNegArrayBefore"] = self.computeFullLogNeg(inState=True)
                    self.saveData(plotsDataDirectory, results["logNegArrayBefore"], TypeOfData.FullLogNegBefore, date)

                elif computation == TypeOfData.OneByOneForAGivenMode:
                    if self.plottingInfo["NumberOfStates"] == 1:
                        oneToOneGivenModes = dict()
                        oneToOneGivenModes[1] = np.zeros((len(specialModes), self.MODES))
                        for index, mode in enumerate(specialModes):
                            oneToOneGivenModes[1][index] = self.computeOneByOneForAGivenMode(mode)[1]
                        results["oneToOneGivenModes"] = oneToOneGivenModes
                    else:
                        print("For more than one initial state OneByOne for a list of modes is not computed")

                elif computation == TypeOfData.OneVSTwoForAGivenMode:
                    if self.plottingInfo["NumberOfStates"] == 1:
                        oneVsTwoGivenModes = dict()
                        oneVsTwoGivenModes[1] = np.zeros((len(specialModes), comb(self.MODES - 1, 2)))
                        for index, mode in enumerate(specialModes):
                            oneVsTwoGivenModes[1][index] = self.computeOneVSTwoForAGivenMode(mode)[1]
                        results["oneVSTwoForAGivenModes"] = oneVsTwoGivenModes
                    else:
                        print("For more than one initial state OneVsTwo for a list of modes is not computed")


                elif computation == TypeOfData.JustSomeModes:
                    results["justSomeModes"] = self.computeFullLogNeg(specialModes=specialModes)

                elif computation == TypeOfData.HawkingPartner:
                    if results["logNegArray"] is None:
                        results["logNegArray"] = self.computeFullLogNeg(specialModes=specialModes)

                    results["hawkingPartner"] = self.computeHawkingPartnerLogNeg(specialModes=specialModes)
        return results

    def plotFullLogNeg(self, logNegArray, plotsDirectory, saveFig=True, specialModes=None):
        if logNegArray is not None:
            if specialModes is not None:
                karray = [idx + 1 for idx in specialModes]
                numberOfModes = len(specialModes)
            else:
                karray = self.kArray
                numberOfModes = self.MODES

            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LN${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                   self.plottingInfo["Magnitude"][index],
                                                   self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None
                pl.loglog(karray, logNegArray[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(numberOfModes / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.FullLogNeg, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotHighestOneByOne(self, highestOneToOneValue, highestOneToOnePartner, logNegArray, plotsDirectory,
                            saveFig=True):
        if highestOneToOneValue is not None and highestOneToOnePartner is not None:
            if logNegArray is None:
                logNegArray = self.computeFullLogNeg()
            pl.figure(figsize=(12, 6))
            pl.loglog(self.kArray[:], highestOneToOneValue, label=r"Strongest one to one $LN$", alpha=0.5, marker='.',
                      markersize=8, linewidth=0.2)
            if logNegArray is not None:
                pl.loglog(self.kArray[:], logNegArray[1][:], label=r"Full $LN$", alpha=0.5, marker='.', markersize=8,
                          linewidth=0.2)

            y_values_highest = np.concatenate(
                (highestOneToOneValue, logNegArray[1][:] if logNegArray is not None else []))
            y_values_Full = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_values = np.concatenate([y_values_highest, y_values_Full])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            if len(self.kArray) == len(highestOneToOneValue) == len(highestOneToOnePartner):
                for i, txt in enumerate(highestOneToOnePartner):
                    pl.annotate(txt + 1, (self.kArray[i], highestOneToOneValue[i]), textcoords="offset points",
                                xytext=(0, 10), ha='center')
            else:
                raise ValueError("The lengths of k_array, maxValues, and maxPartners do not match.")

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.HighestOneByOne, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotOccupationNumber(self, occupationNumber, plotsDirectory, saveFig=True):
        if occupationNumber is not None:
            pl.figure(figsize=(12, 6))
            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$n${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                  self.plottingInfo["Magnitude"][index],
                                                  self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None
                pl.loglog(self.kArray[:], occupationNumber[index + 1], label=label, alpha=0.5, marker='.', markersize=8,
                          linewidth=0.2)

            y_values = np.concatenate(
                [occupationNumber[index + 1] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$n$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OccupationNumber, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotOddVsEven(self, logNegEvenVsOdd, logNegArray, plotsDirectory, saveFig=True):
        if logNegEvenVsOdd is not None:
            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LN$ Odd vs Even{}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                               self.plottingInfo["Magnitude"][index],
                                                               self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "$LN$ Odd vs Even"
                pl.loglog(self.kArray[:], logNegEvenVsOdd[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values_EvenOdd = np.concatenate(
                [logNegEvenVsOdd[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])

            if logNegArray is not None:
                if self.plottingInfo["NumberOfStates"] == 1:
                    pl.loglog(self.kArray[:], logNegArray[1][:], label=r"Full $LN$", alpha=0.5, marker='.',
                              markersize=8, linewidth=0.2)

                y_values_Full = np.concatenate(
                    [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
                y_values = np.concatenate([y_values_EvenOdd, y_values_Full])

            else:
                y_values = y_values_EvenOdd

            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OddVSEven, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotSameParity(self, logNegSameParity, logNegArray, plotsDirectory, saveFig=True):
        if logNegSameParity is not None:
            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LN$ Same Parity {}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                                self.plottingInfo["Magnitude"][index],
                                                                self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "$LN$ Same Parity"
                pl.loglog(self.kArray[:], logNegSameParity[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values_SameParity = np.concatenate(
                [logNegSameParity[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])

            if logNegArray is not None:
                if self.plottingInfo["NumberOfStates"] == 1:
                    pl.loglog(self.kArray[:], logNegArray[1][:], label=r"Full $LN$", alpha=0.5, marker='.',
                              markersize=8, linewidth=0.2)

                y_values_Full = np.concatenate(
                    [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
                y_values = np.concatenate([y_values_SameParity, y_values_Full])

            else:
                y_values = y_values_SameParity

            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.SameParity, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotOneByOneForGivenMode(self, oneToOneGivenModes, specialModes, plotsDirectory, plotsDataDirectory,
                                 saveFig=True, saveData=True):
        if oneToOneGivenModes is not None:
            pl.figure(figsize=(12, 6))

            for index, mode in enumerate(specialModes):
                label = r"$LN$ {} vs each other".format(mode)
                pl.loglog(self.kArray[:], oneToOneGivenModes[1][index][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values = np.concatenate(
                [oneToOneGivenModes[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"] + "{}${: .2f}{}$".format(
                    self.plottingInfo["MagnitudeName"], self.plottingInfo["Magnitude"][0],
                    self.plottingInfo["MagnitudeUnits"]) if self.plottingInfo["Magnitude"][
                                                                0] != "" else "", fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OneByOneForAGivenMode, date)
                pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')

            if saveData:
                fileName = self.getFileName(plotsDataDirectory, TypeOfData.OneByOneForAGivenMode, date)[
                    0]  # Asegurarse de que fileName sea una cadena de texto
                dataToSave = np.zeros((self.MODES + 2, len(specialModes)))
                for index, mode in enumerate(specialModes):
                    arrayParameter = 0
                    if self.arrayParameters is not None:
                        arrayParameter = self.arrayParameters[0]
                    dataToSave[0, index] = arrayParameter
                    dataToSave[1, index] = mode
                    dataToSave[2:, index] = oneToOneGivenModes[1][index]

                os.makedirs(os.path.dirname(fileName), exist_ok=True)

                for index, mode in enumerate(specialModes):
                    np.savetxt("{}_plotNumber_{}.txt".format(fileName, index + 1), dataToSave[:, index])

    def plotOneVsTwoForAGivenMode(self, oneVsTwoForGivenMode, specialModes, plotsDirectory,
                                  plotsDataDirectory, saveFig=True, saveData=True):
        if oneVsTwoForGivenMode is not None and len(specialModes) == 1:
            print(oneVsTwoForGivenMode)

            allPairs = [(i, j) for i, j in combinations(range(self.MODES), 2)
                        if specialModes[0] - 1 not in (i, j)]
            mode = specialModes[0]
            dataVector = oneVsTwoForGivenMode[1][0]
            matrix = np.full((self.MODES, self.MODES), np.nan)

            for index, (i, j) in enumerate(allPairs):
                value = dataVector[index]
                scalarValue = float(value) if np.ndim(value) != 0 else value
                matrix[i, j] = scalarValue
                matrix[j, i] = scalarValue

            # Mask lower triangle
            mask = np.tril(np.ones_like(matrix, dtype=bool))

            # Create figure
            pl.figure(figsize=(8, 6))

            # Validar los datos antes de calcular vmin y vmax
            valid_data = matrix[~np.isnan(matrix) & (matrix > 0)]
            if valid_data.size > 0:
                vmin = np.min(valid_data)
                vmax = np.max(valid_data)
            else:
                vmin = 1e-10  # Valor predeterminado para evitar errores
                vmax = 1  # Valor predeterminado para evitar errores

            # Log normalization para valores positivos
            norm = LogNorm(vmin=vmin, vmax=vmax)

            # Heatmap sin etiquetas automáticas
            sns.heatmap(matrix, mask=mask, cmap='viridis', norm=norm, square=True,
                        cbar_kws={'label': r"$LogNeg$ (log scale)"}, rasterized=False,
                        xticklabels=False, yticklabels=False)

            # Apply hardcoded limits if requested
            xMin, xMax, yMin, yMax = 1, self.MODES, 1, self.MODES
            applyLimits = True
            if applyLimits:
                xMin, xMax = 0, 50
                yMin, yMax = 0, 50
                pl.xlim(xMin, xMax)
                pl.ylim(yMin, yMax)

            # Set custom ticks espaciados
            stepX = max(1, xMax // 10)
            stepY = max(1, yMax // 10)
            ticksX = np.arange(0, xMax, stepX)
            ticksY = np.arange(0, yMax, stepY)
            tick_labels = ticksX + 1
            pl.xticks(ticksX + 0.5, tick_labels, rotation=0)
            pl.yticks(ticksY + 0.5, tick_labels, rotation=0)

            pl.title(f"LogNeg of mode {mode + 1} vs all pairs (excluding {mode + 1})" + "{}${: .2f}{}$".format(
                self.plottingInfo["MagnitudeName"], self.plottingInfo["Magnitude"][0],
                self.plottingInfo["MagnitudeUnits"]) if self.plottingInfo["Magnitude"][
                                                            0] != "" else "")
            pl.xlabel("Mode i")
            pl.ylabel("Mode j")

            # Add top 3 points if enabled
            applyPoints = True
            if applyPoints:
                maskedMatrix = np.triu(matrix, k=1)
                validIndices = [(i, j) for i in range(maskedMatrix.shape[0])
                                for j in range(maskedMatrix.shape[1])
                                if not np.isnan(maskedMatrix[i, j]) and
                                (not applyLimits or (xMin <= j < xMax and yMin <= i < yMax))]

                topCoords = sorted(validIndices, key=lambda x: maskedMatrix[x[0], x[1]], reverse=True)[:3]

                colors = ['red', 'blue', 'green']
                handles = []

                for idx, (i, j) in enumerate(topCoords):
                    color = colors[idx % len(colors)]
                    # Mostrar índices 1-based en la leyenda
                    handle = pl.Line2D([0], [0], marker='o', color='w', label=f'({i + 1},{j + 1})',
                                       markerfacecolor=color, markersize=8)
                    handles.append(handle)
                    pl.plot(j + 0.5, i + 0.5, 'o', color=color)

                pl.legend(handles=handles, title="Top LogNeg (i,j)", loc='upper right', fontsize=8, title_fontsize=9)

            pl.tight_layout()

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OneVSTwoForAGivenMode, date)
                pl.savefig(figureName)

            if saveData:
                fileNameBase = self.getFileName(plotsDataDirectory, TypeOfData.OneVSTwoForAGivenMode, date)[0]
                os.makedirs(os.path.dirname(fileNameBase), exist_ok=True)
                np.savetxt(f"{fileNameBase}_mode_{mode}.txt", matrix)

            pl.close()

    def plotLogNegDifference(self, differenceArray, logNegArray, logNegArrayBefore, plotsDirectory, saveFig=True):
        if differenceArray is not None:
            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LNAfter-LNBefore${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                                 self.plottingInfo["Magnitude"][index],
                                                                 self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None
                pl.loglog(self.kArray[:], differenceArray[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

                labelLogNeg = r"$LNAfter${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                              self.plottingInfo["Magnitude"][index],
                                                              self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None

                pl.loglog(self.kArray[:], logNegArray[index + 1][:], label=labelLogNeg, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

                labelLogNegBefore = r"$LNBefore${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                                     self.plottingInfo["Magnitude"][index],
                                                                     self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None

                pl.loglog(self.kArray[:], logNegArrayBefore[index + 1][:], label=labelLogNegBefore, alpha=0.5,
                          marker='.', markersize=8, linewidth=0.2)

            y_values_logNegAfter = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_values_logNegBefore = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])

            y_min = np.min([np.min(y_values_logNegAfter), np.min(y_values_logNegBefore)])
            y_max = np.max([np.max(y_values_logNegAfter), np.max(y_values_logNegBefore)])

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.LogNegDifference, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotHawkingPartner(self, hawkingPartnerArray, logNegArray, specialModes, plotsDirectory, saveFig=True):
        if logNegArray is not None and hawkingPartnerArray is not None:
            karray = [idx + 1 for idx in specialModes]
            numberOfModes = len(specialModes)

            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                labelFull = r"$FullLN${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                           self.plottingInfo["Magnitude"][index],
                                                           self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "FullLN"

                labelHP = r"$Hawk-P LN${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                            self.plottingInfo["Magnitude"][index],
                                                            self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "Hawk-P LN"

                pl.loglog(karray, logNegArray[index + 1][:], label=labelFull, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)
                pl.loglog(karray, hawkingPartnerArray[index + 1][:], label=labelHP, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values_fullLog = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min_fl = np.min(y_values_fullLog)
            y_max_fl = np.max(y_values_fullLog)

            y_values_hp = np.concatenate(
                [hawkingPartnerArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min_hp = np.min(y_values_hp)
            y_max_hp = np.max(y_values_hp)

            y_min = np.min([y_min_hp, y_min_fl])
            y_max = np.max([y_max_hp, y_max_fl])

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(numberOfModes / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if labelFull is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.FullLogNeg, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def generatePlots(self, results, plotsDirectory, plotsDataDirectory, specialModes, listOfWantedComputations,
                      saveFig=True):
        logNegArray = results.get("logNegArray")
        logNegArrayBefore = results.get("logNegArrayBefore")
        highestOneToOneValue = results.get("highestOneToOneValue")
        highestOneToOnePartner = results.get("highestOneToOnePartner")
        occupationNumber = results.get("occupationNumber")
        logNegEvenVsOdd = results.get("logNegEvenVsOdd")
        logNegSameParity = results.get("logNegSameParity")
        oneToOneGivenModes = results.get("oneToOneGivenModes")
        oneVSTwoForAGivenModes = results.get("oneVSTwoForAGivenModes")
        differenceArray = results.get("logNegDifference")
        justSomeModes = results.get("justSomeModes")
        hawkingPartner = results.get("hawkingPartner")

        if TypeOfData.FullLogNeg in listOfWantedComputations and logNegArray is not None:
            self.plotFullLogNeg(logNegArray, plotsDirectory, saveFig=saveFig)

        if TypeOfData.OccupationNumber in listOfWantedComputations and occupationNumber is not None:
            self.plotOccupationNumber(occupationNumber, plotsDirectory, saveFig=saveFig)

        if TypeOfData.OddVSEven in listOfWantedComputations and logNegEvenVsOdd is not None:
            self.plotOddVsEven(logNegEvenVsOdd, None, plotsDirectory, saveFig=saveFig)

        if TypeOfData.SameParity in listOfWantedComputations and logNegEvenVsOdd is not None:
            self.plotSameParity(logNegSameParity, None, plotsDirectory, saveFig=saveFig)

        if TypeOfData.OneByOneForAGivenMode in listOfWantedComputations and oneToOneGivenModes is not None:
            self.plotOneByOneForGivenMode(oneToOneGivenModes, specialModes, plotsDirectory, plotsDataDirectory,
                                          saveFig=saveFig, saveData=True)

        if TypeOfData.OneVSTwoForAGivenMode in listOfWantedComputations and oneVSTwoForAGivenModes is not None:
            self.plotOneVsTwoForAGivenMode(oneVSTwoForAGivenModes, specialModes, plotsDirectory, plotsDataDirectory,
                                           saveFig=saveFig, saveData=True)

        if TypeOfData.LogNegDifference in listOfWantedComputations and differenceArray is not None:
            if logNegArray is not None and logNegArrayBefore is not None:
                self.plotLogNegDifference(differenceArray, logNegArray, logNegArrayBefore, plotsDirectory,
                                          saveFig=saveFig)

        if TypeOfData.HighestOneByOne in listOfWantedComputations and highestOneToOnePartner is not None and highestOneToOneValue is not None:
            self.plotHighestOneByOne(highestOneToOneValue, highestOneToOnePartner, logNegArray, plotsDirectory,
                                     saveFig=saveFig)

        if TypeOfData.JustSomeModes in listOfWantedComputations and justSomeModes is not None:
            self.plotFullLogNeg(justSomeModes, plotsDirectory, saveFig=saveFig, specialModes=specialModes)

        if TypeOfData.HawkingPartner in listOfWantedComputations and hawkingPartner is not None:
            self.plotHawkingPartner(hawkingPartner, logNegArray, specialModes, plotsDirectory, saveFig=saveFig)

    def obtainHawkingPartner(self, modeA: int = 0, inState: bool = False, atol: float = 1e-8):
        """
        Extracts the basis transformation that maps the original OUT basis to a new one
        where mode 0 is the Hawking mode, mode 1 is the partner, and the rest are reordered.

        Args:
            modeA: mode to be considered the Hawking one
            inState: wheter to obtain the partner for the inState or the outState

        Returns:
             newBogoliubovTransformation: Bogoliubov transformation in the new ordered basis from IN basis -> new OUT basis
             changeOfBasis: 2N x 2N transformation from the original OUT basis → new OUT basis
             statetoInBasis: Gaussian State in the in Basis to perform the newBogo on it (it works for states with thermal noise)
        """

        if self.plottingInfo["NumberOfStates"] != 1:
            raise ValueError("Hawking-partner calculation only works for 1 simulation at once")

        if InitialState.Thermal.value == self.plottingInfo["InStateName"]:
            SInitial = np.eye(2 * self.MODES)
            statetoInBasis = self.inState[1]

        elif InitialState.OneModeSqueezedFixedTemp.value == self.plottingInfo["InStateName"]:
            intensity_array = [self.arrayParameters[0] for _ in range(0, self.MODES)]
            auxiliarInitialState = qgt.elementary_states("squeezed", intensity_array)
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(auxiliarInitialState.V), 0)
            temperature = 0.694554 * self._temperature  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
            n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                        range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)]
            statetoInBasis = qgt.elementary_states("thermal", n_vector)

        elif InitialState.TwoModeSqueezedFixedTemp.value == self.plottingInfo["InStateName"]:
            auxiliarInitialState = qgt.Gaussian_state("vacuum", self.MODES)
            for j in range(0, self.MODES, 2):
                auxiliarInitialState.two_mode_squeezing(self.arrayParameters[0], 0, [j, j + 1])
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(auxiliarInitialState.V), 0)
            temperature = 0.694554 * self._temperature  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
            n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                        range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)]
            statetoInBasis = qgt.elementary_states("thermal", n_vector)

        elif InitialState.ThermalFixedOneModeSqueezing.value == self.plottingInfo["InStateName"]:
            intensity_array = [self._squeezing for _ in range(0, self.MODES)]
            auxiliarInitialState = qgt.elementary_states("squeezed", intensity_array)
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(auxiliarInitialState.V), 0)
            temperature = 0.694554 * self.arrayParameters[
                0]  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
            n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                        range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)]
            statetoInBasis = qgt.elementary_states("thermal", n_vector)

        else:
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(self.inState[1].V), 0)
            statetoInBasis = qgt.Gaussian_state("vacuum", self.MODES)

        if not inState:
            bogoliubovTransformation = self.transformationMatrix @ SInitial
        else:
            bogoliubovTransformation = SInitial

        newBogoliubovTransformation, changeOfBasis = extractHawkingPartner(bogoliubovTransformation, modeA, atol=atol)

        return newBogoliubovTransformation, changeOfBasis, statetoInBasis

    def obtainHPLogNegForListOfModes(self, specialModes: List[int], inState: bool = False, atol: float = 1e-8):
        def makeHPLogNegTask(mode):
            def task():
                newBogoTrans, _, newOutState = self.obtainHawkingPartner(mode, inState=inState)
                newOutState.apply_Bogoliubov_unitary(newBogoTrans)
                logNegPartner = newOutState.logarithmic_negativity([0], [1])
                return mode, logNegPartner

            return task

        tasks = [makeHPLogNegTask(mode) for mode in specialModes]
        return self._execute(tasks)
