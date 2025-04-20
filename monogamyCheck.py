import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from LogNegManager import InitialState, TypeOfData

# === Configurations ===
baseDir = Path("./path/to/plotsData")
plotsDir = Path("./path/to/figures/")
inState = InitialState.TwoModeSqueezedFixedTemp.value
numTemperatures = 3
instant = 13

# === Date range (inclusive) ===
startDate = datetime.strptime("2025-04-18", "%Y-%m-%d")
endDate = datetime.strptime("2025-04-20", "%Y-%m-%d")

# === Optional extra parameters ===
rManual = 0.75  # Only relevant if inState == "thermalFixedOneModeSqueezing"
tManual = 20  # Only relevant if inState == "oneModeSqueezedFixedTemp" or "TwoModeSqueezedFixedTemp"

logNegTypes = [TypeOfData.FullLogNeg.value,
               TypeOfData.OddVSEven.value,
               TypeOfData.SameParity.value]
colors = {
    TypeOfData.FullLogNeg.value: "blue",
    TypeOfData.OddVSEven.value: "green",
    TypeOfData.SameParity.value: "red"
}
labels = {
    TypeOfData.FullLogNeg.value: "Full",
    TypeOfData.OddVSEven.value: "Odd vs Even",
    TypeOfData.SameParity.value: "Same Parity"
}

# === State types for parameter classification ===
statesWithTemperature = {
    InitialState.Thermal.value,
    InitialState.ThermalFixedOneModeSqueezing.value
}
statesWithSqueezing = {
    InitialState.OneModeSqueezed.value,
    InitialState.TwoModeSqueezed.value,
    InitialState.OneModeSqueezedFixedTemp.value,
    InitialState.TwoModeSqueezedFixedTemp.value
}

# === Helper function to filter by date ===
def isDateInRange(fileName):
    try:
        dateStr = fileName.name.split("_date_")[1].split("_plotNumber_")[0]
        date = datetime.strptime(dateStr.split("_")[0], "%Y-%m-%d")
        return startDate <= date <= endDate
    except Exception:
        return False

# === Load data ===
data = {key: [] for key in logNegTypes}
paramValues = []
simulationName = f"{inState}_instant_{instant}_numOfPlots_{numTemperatures}"

# === Determine whether parameter is T or r ===
paramLabel = "T [K]" if inState in statesWithTemperature else "r"

for i in range(1, numTemperatures + 1):
    files = list(baseDir.glob(f"{TypeOfData.FullLogNeg.value}_{simulationName}_date_*_plotNumber_{i}.txt"))
    filePath = [f for f in files if isDateInRange(f)][0]
    arr = np.loadtxt(filePath)
    paramValues.append(arr[0])  # First row: T or r
    data[TypeOfData.FullLogNeg.value].append(arr[1:])  # Rest: LogNeg per mode

# Load oddVSEven and sameParity
for key in [TypeOfData.OddVSEven.value, TypeOfData.SameParity.value]:
    for i in range(1, numTemperatures + 1):
        files = list(baseDir.glob(f"{key}_{simulationName}_date_*_plotNumber_{i}.txt"))
        filePath = [f for f in files if isDateInRange(f)][0]
        arr = np.loadtxt(filePath)
        data[key].append(arr[1:])  # LogNeg from mode 1

# === Plot per parameter value (T or r) ===
for i in range(numTemperatures):
    modes = np.arange(1, len(data[TypeOfData.FullLogNeg.value][i]) + 1)
    for j in range(len(modes)):
        print(f"Mode: {modes[j]}, FullLogNeg: {data[TypeOfData.FullLogNeg.value][i][j]}, "
              f"OddVSEven: {data[TypeOfData.OddVSEven.value][i][j]}, "
              f"SameParity: {data[TypeOfData.SameParity.value][i][j]}")

    plt.figure(figsize=(10, 6))
    for key in logNegTypes:
        plt.plot(modes, data[key][i], label=labels[key], color=colors[key])

    residual = data[TypeOfData.FullLogNeg.value][i] - (
        data[TypeOfData.OddVSEven.value][i] + data[TypeOfData.SameParity.value][i]
    )
    plt.plot(modes, residual, label="Residual: Full - (Odd + Same)", linestyle='--', color='black')

    paramVal = paramValues[i]
    paramStr = f"{paramVal:.2f}".replace(".", "p")

    # === Add extra info to title and filename if needed ===
    extraLabel = ""
    extraFilename = ""
    if inState == InitialState.ThermalFixedOneModeSqueezing.value:
        extraLabel = f", r = {rManual:.2f}"
        extraFilename = f"_r_{str(rManual).replace('.', 'p')}"
    elif inState in {
        InitialState.OneModeSqueezedFixedTemp.value,
        InitialState.TwoModeSqueezedFixedTemp.value
    }:
        extraLabel = f", T = {tManual:.3f} K"
        extraFilename = f"_T_{str(tManual).replace('.', 'p')}K"

    residualPath = plotsDir / f"residual_{paramLabel.replace(' ', '')}_{paramStr}{extraFilename}_{simulationName}.txt"
    np.savetxt(residualPath, np.column_stack((modes, residual)), header="ModeIndex Residual")

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"LogNeg comparison at {paramLabel} = {paramVal:.2f}{extraLabel}")
    plt.xlabel("Mode index (log scale)")
    plt.ylabel("Logarithmic Negativity (log scale)")
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()

    plt.savefig(plotsDir / f"loglog_logneg_comparison_{paramLabel.replace(' ', '')}_{paramStr}{extraFilename}_{simulationName}.pdf")
    plt.close()




