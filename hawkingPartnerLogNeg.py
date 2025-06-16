import os
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

from LogNegManager import InitialState, LogNegManager, TypeOfData


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


def compute_logneg_arrays(simulation, modesToApply, plotsDataDirectory):
    logNegArrayResults = simulation.performComputations(
        [TypeOfData.JustSomeModes], plotsDataDirectory, tryToLoad=False, specialModes=modesToApply)
    fullLogNegArray = logNegArrayResults["justSomeModes"][1]
    resultsHPLN = simulation.obtainHPLogNegForListOfModes(modesToApply)
    HPLogNegArray = np.zeros(len(modesToApply))
    for mode, logNeg in resultsHPLN:
        HPLogNegArray[modesToApply.index(mode)] = logNeg
    return fullLogNegArray, HPLogNegArray


def save_npz_per_squeezing(plotsDataDirectory, squeezingFactor, temperature, inStateType, modesToApply, fullLogNegArray,
                           HPLogNegArray, date):
    npz_filename = os.path.join(
        plotsDataDirectory,
        f"HPvsFull-{date}.npz"
    )
    np.savez(
        npz_filename,
        squeezingFactor=squeezingFactor,
        temperature=temperature,
        modesToApply=np.array(modesToApply),
        inStateType=inStateType.value,
        FullLogNegArray=np.array(fullLogNegArray),
        HPLogNegArray=np.array(HPLogNegArray)
    )


def plot_logneg(karray, fullLogNegArray, HPLogNegArray, squeezingFactor, temperature, inStateType, numberOfModes,
                plotsDirectory, date):
    plt.figure(figsize=(12, 6))
    labelFull = "FullLN"
    labelHP = "Hawk-P LN"
    plt.loglog(karray, fullLogNegArray, label=labelFull, alpha=0.5, marker='.', markersize=8, linewidth=0.2)
    plt.loglog(karray, HPLogNegArray, label=labelHP, alpha=0.5, marker='.', markersize=8, linewidth=0.2)

    y_values = np.concatenate([fullLogNegArray, HPLogNegArray])
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

    plt.xlim(1, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(r"$I$", fontsize=20)
    plt.ylabel(r"$LogNeg(I)$", fontsize=20)
    plt.grid(linestyle="--", color='0.9')
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
    rc('xtick', labelsize=16)
    rc('ytick', labelsize=16)
    plt.tight_layout()
    if inStateType.value == InitialState.OneModeSqueezedFixedTemp.value:
        plt.suptitle(f"One Mode Squeezing: {squeezingFactor}, temperature: {temperature}", fontsize=20)
    elif inStateType.value == InitialState.TwoModeSqueezedFixedTemp.value:
        plt.suptitle(f"Two Mode Squeezing: {squeezingFactor}, temperature: {temperature}", fontsize=20)
    figureName = f"HPvsFull-{date}.pdf"
    plt.savefig(os.path.join(plotsDirectory, figureName), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()


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

    MODES = 128
    inStateType = InitialState.TwoModeSqueezedFixedTemp  # Choose between OneMoOneModeSqueezedFixedTemp or TwoModeSqueezedFixedTemp
    squeezingFactors = [0.0, 0.5, 1.0]
    temperatures = [0.0, 5.0, 10.0]
    totalModes = 15
    parallelize = True

    dataDirectory, plotsDirectory, plotsDataDirectory = get_directories(MODES, dictNumberModes)
    modesToApply = [idx for idx in range(totalModes)]

    start_time = time.time()

    for squeezingFactor in squeezingFactors:
        for temperature in temperatures:
            if inStateType not in [InitialState.OneModeSqueezedFixedTemp, InitialState.TwoModeSqueezedFixedTemp]:
                raise AttributeError(
                    "You have to select between between OneMoOneModeSqueezedFixedTemp or TwoModeSqueezedFixedTemp")

            simulation = LogNegManager(
                dataDirectory, inStateType, MODES, instantToPlot=1000,
                arrayParameters=np.array([squeezingFactor]), temperature=temperature,
                parallelize=parallelize
            )

            simulation.performTransformation()
            fullLogNegArray, HPLogNegArray = compute_logneg_arrays(simulation, modesToApply, plotsDataDirectory)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_npz_per_squeezing(plotsDataDirectory, squeezingFactor, temperature, inStateType, modesToApply,
                                   fullLogNegArray, HPLogNegArray, date)
            karray = [idx + 1 for idx in modesToApply]
            numberOfModes = len(modesToApply)
            plot_logneg(karray, fullLogNegArray, HPLogNegArray, squeezingFactor, temperature, inStateType,
                        numberOfModes, plotsDirectory, date)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total computation time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
