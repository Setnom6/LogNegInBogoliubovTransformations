import os
import time
from datetime import datetime

import matplotlib.pyplot as plt

import qgt
from LogNegManager import LogNegManager, InitialState
from partnerMethods import *


def createTestTransformation(totalModes):
    """
    Creates a test transformation for a given number of modes.

    Parameters:
        totalModes: total number of modes

    Returns:
        state: Gaussian state
        bogoliubov_inv: inverse Bogoliubov transformation matrix
    """
    state = Gaussian_state("squeezed", totalModes, 1.0)

    Sinitial = BasisChange(get_symplectic_from_covariance(state.V), 0)
    TMS1 = state.two_mode_squeezing(r=1.0, phi=0.0, modes=[2, 3])
    # TMS2 = state.two_mode_squeezing(r=0.5, phi=-12.0, modes=[0, 1])
    # TMS3 = state.two_mode_squeezing(r=1.0, phi=3.0, modes=[2, 3])
    # BS = state.beam_splitter(tau=0.75, modes=[0, 3])
    # BS2 = state.beam_splitter(tau=0.5, modes=[0, 1])

    # U = BS2 @ BS @ TMS3 @ TMS2 @ TMS1 @ Sinitial

    U = TMS1 @ Sinitial
    bogoliubov = BasisChange(U, 0)
    omega = state.Omega
    bogoliubov_inv = np.linalg.inv(omega) @ bogoliubov.conj().T @ omega

    return state, bogoliubov_inv


def createSimulationWithSpecificInitialState(totalModes, dataDirectory, temperature, squeezingOne, parallelize):
    """
    Creates a simulation with a specific initial state.

    Parameters:
        totalModes: total number of modes
        dataDirectory: directory for storing data
        temperature: temperature of the initial state
        squeezingOne: squeezing parameter
        parallelize: whether to parallelize the computation

    Returns:
        LogNegManager instance
    """
    instant = 1000

    arrayParameters = np.array([squeezingOne])
    if squeezingOne == 0.0 and temperature == 0.0:
        inStateType = InitialState.Vacuum
    elif squeezingOne != 0.0 and temperature == 0.0:
        inStateType = InitialState.OneModeSqueezed
    elif squeezingOne == 0.0 and temperature != 0.0:
        inStateType = InitialState.Thermal
        arrayParameters = np.array([temperature])
    else:
        inStateType = InitialState.OneModeSqueezedFixedTemp

    return LogNegManager(
        dataDirectory,
        inStateType,
        totalModes,
        instant,
        arrayParameters,
        temperature=temperature,
        parallelize=parallelize,
    )


def createRealTransformation(totalModes, squeezingOne=0.0, parallelize=False, dictNumberModes=None):
    """
    Obtains the transformation that maps the original (in) basis to the Hawking-Partner-extra modes basis
    for a pure initial state (vacuum or oneModeSqueezed).

    Parameters:
        totalModes: total number of modes
        squeezingOne: squeezing parameter
        parallelize: whether to parallelize the computation

    Returns:
        state: Gaussian state
        bogoliubov: Bogoliubov transformation matrix
        dataDirectory: directory for storing data
        plotsDirectory: directory for storing plots
        dataPlotsDirectory: directory for storing plot data
    """
    if totalModes in dictNumberModes.keys():
        dataDirectory = dictNumberModes[totalModes]['dataDirectory']
        plotsDirectory = dictNumberModes[totalModes]['plotsDirectory']
        dataPlotsDirectory = dictNumberModes[totalModes]['dataPlotsDirectory']
    else:
        raise NotImplementedError(f"Total modes for {totalModes} not implemented.")

    simulation = createSimulationWithSpecificInitialState(totalModes, dataDirectory,
                                                          temperature=0.0, squeezingOne=squeezingOne,
                                                          parallelize=parallelize)

    simulation.performTransformation()
    state = simulation.outState[1]

    SInitial = np.eye(2 * totalModes, dtype=complex)
    if squeezingOne != 0.0:
        # If the initial state is not the Vacuum, recover the first transformation as the algorithm assumes
        # that the initial state is the vacuum
        inState = simulation.inState[1]
        SInitial = BasisChange(get_symplectic_from_covariance(inState.V), 0)

    bogoliubov = simulation.transformationMatrix @ SInitial

    return state, bogoliubov, dataDirectory, plotsDirectory, dataPlotsDirectory


def getLogNegValuesPerTemperature(totalModes, modeA, squeezing, temperatures, newBogoliubov, dataDirectory,
                                  parallelize):
    """
    Computes the logarithmic negativity for different temperatures.

    Parameters:
        totalModes: total number of modes
        modeA: mode index
        squeezing: squeezing parameter
        temperatures: list of temperatures
        newBogoliubov: Bogoliubov transformation matrix
        dataDirectory: directory for storing data
        parallelize: whether to parallelize the computation

    Returns:
        logNegOriginalList: list of logarithmic negativities in the original basis
        logNegPartnerList: list of logarithmic negativities in the Hawking-Partner basis
    """
    logNegOriginalList = []
    logNegPartnerList = []

    for temp in temperatures:
        simulationOriginalBasis = createSimulationWithSpecificInitialState(
            totalModes=totalModes,
            dataDirectory=dataDirectory,
            temperature=temp,
            squeezingOne=squeezing,
            parallelize=parallelize
        )

        simulationOriginalBasis.performTransformation()
        outStateOriginalBasis = simulationOriginalBasis.outState[1].copy()
        logNegOriginal = outStateOriginalBasis.logarithmic_negativity([modeA],
                                                                      [i for i in range(totalModes) if i != modeA])
        logNegOriginalList.append(logNegOriginal)

        # Transform to Hawking-Partner basis
        newState = simulationOriginalBasis.inState[1].copy()
        newState = qgt.Gaussian_state("vacuum", totalModes)
        newState.apply_Bogoliubov_unitary(newBogoliubov)

        logNegPartner = newState.logarithmic_negativity([0], [1])
        logNegPartnerList.append(logNegPartner)

        print(f"temp {temp} original: ", logNegOriginal)
        print(f"changed: ", logNegPartner)
        print("\n")

    return logNegOriginalList, logNegPartnerList


def plotPartnerContributions(HPExpressedInOUTBasis, totalModes, modeA, plotsDirectory, dataPlotsDirectory, squeezing,
                             thresholdRatio=1e-3):
    """
    Plots the contributions of the partner mode in the OUT basis.

    Parameters:
        HPExpressedInOUTBasis: transformation matrix from OUT basis to Hawking-Partner basis
        totalModes: total number of modes
        modeA: mode index
        plotsDirectory: directory for storing plots
        dataPlotsDirectory: directory for storing plot data
        squeezing: squeezing parameter
        thresholdRatio: threshold for considering contributions negligible

    Returns:
        partnerContributions: array of contributions for each mode
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plotsPath = None
    if plotsDirectory is not None:
        plotsPath = os.path.join(plotsDirectory, f"partnerContributions_vacuum_{timestamp}.png")

    dataPath = None
    if dataPlotsDirectory is not None:
        dataPath = os.path.join(dataPlotsDirectory, f"partnerContributions_vacuum_{timestamp}.npz")

    partnerVector = HPExpressedInOUTBasis[2]
    partnerContributions = np.array([
        abs(partnerVector[2 * i]) ** 2 + abs(partnerVector[2 * i + 1]) ** 2
        for i in range(totalModes)
    ])

    maxContribution = np.max(partnerContributions)
    modeAContribution = partnerContributions[modeA]

    if modeAContribution < thresholdRatio * maxContribution:
        print(
            f"Contribution from modeA={modeA + 1} is negligible: {modeAContribution:.2e} vs max {maxContribution:.2e}")
        plotContributions = np.delete(partnerContributions, modeA)
        kArray = np.delete(np.arange(totalModes), modeA)
    else:
        print(f"Warning: modeA={modeA + 1} has a significant contribution: {modeAContribution:.2e}")
        plotContributions = partnerContributions
        kArray = np.arange(totalModes)

    kArray = [i + 1 for i in kArray]  # Convert to 1-based index

    # Guardar datos
    if dataPath is not None:
        np.savez(dataPath,
                 modeIndices=np.array(kArray),
                 contributions=np.array(plotContributions),
                 modeA=modeA + 1,
                 squeezing=squeezing,
                 maxContribution=maxContribution,
                 modeAContribution=modeAContribution)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.loglog(kArray, plotContributions, marker='o', label='Mode contributions')
    plt.axvline(x=modeA + 1, color='red', linestyle='--', label=f'modeA = {modeA + 1}')
    plt.xlabel("Mode index $k$")
    plt.ylabel("Contribution to partner mode")
    plt.title("Partner mode contributions by OUT modes \n"
              f"OneModeSqz: {squeezing}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    if plotsPath is not None:
        plt.savefig(plotsPath)
    plt.show()

    return partnerContributions


def plotLogNegVsTemperature(temperatures, logNegOriginal, logNegPartner, plotsDirectory, dataPlotsDirectory, squeezing):
    """
    Plots the logarithmic negativity as a function of temperature.

    Parameters:
        temperatures: list of temperatures
        logNegOriginal: logarithmic negativity in the original basis
        logNegPartner: logarithmic negativity in the Hawking-Partner basis
        plotsDirectory: directory for storing plots
        dataPlotsDirectory: directory for storing plot data
        squeezing: squeezing parameter
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plotPath = None
    if plotsDirectory is not None:
        plotPath = os.path.join(plotsDirectory, f"logNeg_vs_temp_squeezing_{squeezing}_{timestamp}.pdf")

    dataPath = None
    if dataPlotsDirectory is not None:
        dataPath = os.path.join(dataPlotsDirectory, f"logNeg_vs_temp_squeezing_{squeezing}_{timestamp}.npz")

    # Guardar datos
    if dataPath is not None:
        np.savez(dataPath,
                 temperatures=np.array(temperatures),
                 logNegOriginal=np.array(logNegOriginal),
                 logNegPartner=np.array(logNegPartner),
                 squeezing=squeezing)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(temperatures, logNegOriginal, marker='o', label='Original basis: $A$ vs Rest')
    plt.plot(temperatures, logNegPartner, marker='s', label='Hawking basis: $H$ vs $P$')
    plt.xlabel("Temperature")
    plt.ylabel("Logarithmic Negativity")
    plt.title(f"Log Neg vs Temperature\nSqueezing = {squeezing}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if plotPath is not None:
        plt.savefig(plotPath)
    plt.show()


def obtainHawkingPartnerAndPlotResults(totalModes=256, modeA=1, squeezing=0.0, temperatures=None,
                                       parallelize=False, plotContributions=True, atol=1.0e-8, dictNumberModes=None):
    """
    Computes the Hawking-Partner transformation and plots the results.

    Parameters:
        totalModes: total number of modes
        modeA: mode index
        squeezing: squeezing parameter
        temperatures: list of temperatures
        parallelize: whether to parallelize the computation
        plotContributions: whether to plot partner contributions
        atol: tolerance for numerical errors
    """
    modeA -= 1  # Convert to 0-indexed

    if totalModes not in dictNumberModes.keys():
        state, bogoliubov = createTestTransformation(totalModes)
        newBogoliubov, changeOfBasis = extractHawkingPartner(bogoliubov, modeA, atol=atol)
        plotsDirectory = None
        dataPlotsDirectory = None

        if plotContributions:
            plotPartnerContributions(changeOfBasis, totalModes, modeA, plotsDirectory, dataPlotsDirectory, 0.0)

    else:

        state, bogoliubov, dataDirectory, plotsDirectory, dataPlotsDirectory = createRealTransformation(
            totalModes, squeezing, parallelize, dictNumberModes)

        newBogoliubov, changeOfBasis = extractHawkingPartner(bogoliubov, modeA, atol=atol)

        if plotContributions:
            plotPartnerContributions(changeOfBasis, totalModes, modeA, plotsDirectory, dataPlotsDirectory, squeezing)

        if temperatures is None:
            temperatures = [0.0]

        logNegOriginal, logNegPartner = getLogNegValuesPerTemperature(
            totalModes, modeA, squeezing, temperatures, newBogoliubov, dataDirectory, parallelize
        )

        plotLogNegVsTemperature(temperatures, logNegOriginal, logNegPartner, plotsDirectory, dataPlotsDirectory,
                                squeezing)


#################################################################################################
#                                                                                               #
#  Simulacion con 128 o 256 modos. Se puede cambiar la ruta a la carpeta de alphas y betas      #
#  para ello cambiar el dictNumberModes                                                         #
#  El partner mode se puede calcular para un estado inicial puro (vacío o one-mode squeezed)    #
#  Si se da un array de temperaturas se realiza la misma transformacion obtenida para el vacio  #
#  Caso de test se puede hacer con createTestTransformation, meter en la funcion las            #
#  trasnformaciones que se quieran  (modifica createTestTransformation). Tambien se activa      #
#  esa función si el totalModes no está previsto en dictNumberModes.                            #
#                                                                                               #
#  Para trabajar con la nueva relación de Bogoliubov se puede usar la funcion                   #
#  obtainChangeOfBasisAndHawkingPartner que simplemente devuelve la nueva bogoliubov completa   #
#  y el cambio de base del out antiguo al HP                                                    #
#                                                                                               #
#################################################################################################

dictNumberModes = {
    128: {'dataDirectory': "./sims-128/",
          'plotsDirectory': "./plots/128-1plt-plots/",
          'dataPlotsDirectory': "./plotsData/128-1plt-data/"},
    256: {
        'dataDirectory': "./simssquid06-256-1plt-dL0375-k12-5-april/",
        'plotsDirectory': "./plots/256-april/",
        'dataPlotsDirectory': "./plotsData/256-april/"
    }
}

totalModes = 128
modesA = [3]
squeezings = [1.0]
parallelize = True
temperatures = [0.0, 1.0, 5.0, 10.0]

startTime = time.time()

for modeA in modesA:
    for squeezing in squeezings:
        obtainHawkingPartnerAndPlotResults(totalModes, modeA, squeezing, temperatures, parallelize,
                                           plotContributions=True, dictNumberModes=dictNumberModes)

endTime = time.time()

print(f"Tiempo total de ejecución: {endTime - startTime:.2f} segundos")
