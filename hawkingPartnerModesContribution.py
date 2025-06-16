import os
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from LogNegManager import InitialState, LogNegManager


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_directories(MODES, dictNumberModes):
    dataDirectory = dictNumberModes[MODES]['dataDirectory']
    plotsDirectory = dictNumberModes[MODES]['plotsDirectory']
    plotsDataDirectory = dictNumberModes[MODES]['dataPlotsDirectory']
    ensure_directory_exists(dataDirectory)
    ensure_directory_exists(plotsDirectory)
    ensure_directory_exists(plotsDataDirectory)
    return dataDirectory, plotsDirectory, plotsDataDirectory


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
        plotsPath = os.path.join(plotsDirectory, f"partnerContributions_{timestamp}.png")

    dataPath = None
    if dataPlotsDirectory is not None:
        dataPath = os.path.join(dataPlotsDirectory, f"partnerContributions_{timestamp}.npz")

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
                 modeA=modeA,
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


def main():
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

    # Parameters to modify (change also the paths in dictNumberModes)
    MODES = 256
    inStateType = InitialState.TwoModeSqueezed  # Choose between OneMoOneModeSqueezed or TwoModeSqueezed
    squeezingFactors = [0.0, 0.5, 1.0]
    modesToPlot = [1, 3, 27, 100]  # From 1 to MODES
    parallelize = True

    # Loop
    dataDirectory, plotsDirectory, plotsDataDirectory = get_directories(MODES, dictNumberModes)

    start_time = time.time()

    for squeezingFactor in squeezingFactors:
        for modeToPlot in modesToPlot:
            modeToPlot -= 1
            if inStateType not in [InitialState.OneModeSqueezed, InitialState.TwoModeSqueezed]:
                raise AttributeError(
                    "You have to select between between OneMoOneModeSqueezed or TwoModeSqueezed")

            simulation = LogNegManager(
                dataDirectory, inStateType, MODES, instantToPlot=1000,
                arrayParameters=np.array([squeezingFactor]),
                parallelize=parallelize
            )

            simulation.performTransformation()
            newBogoliubovTransformation, changeOfBasis, statetoInBasis = simulation.obtainHawkingPartner(
                modeA=modeToPlot)
            plotPartnerContributions(HPExpressedInOUTBasis=changeOfBasis, totalModes=MODES, modeA=modeToPlot,
                                     plotsDirectory=plotsDirectory, dataPlotsDirectory=plotsDataDirectory,
                                     squeezing=squeezingFactor)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total computation time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
