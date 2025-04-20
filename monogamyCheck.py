import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from LogNegManager import InitialState, TypeOfData

# === Configurations ===
baseDir = Path("./path/to/plotsData")
plotsDir = Path("./path/to/figures/")
inState = InitialState.TwoModeSqueezedFixedTemp.value
numTemperatures = 5
instant = 13

startDate = datetime.strptime("2025-04-18", "%Y-%m-%d")
endDate = datetime.strptime("2025-04-20", "%Y-%m-%d")

# === Extra parameters if needed ===
rManual = 0.75
tManual = 20

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

def isDateInRange(fileName):
    try:
        dateStr = fileName.name.split("_date_")[1].split("_plotNumber_")[0]
        date = datetime.strptime(dateStr.split("_")[0], "%Y-%m-%d")
        return startDate <= date <= endDate
    except Exception:
        return False

def sanitize(val):
    return str(val).replace('.', 'p')

data = {key: [] for key in logNegTypes}
paramValues = []
simulationName = f"{inState}_instant_{instant}_numOfPlots_{numTemperatures}"
paramLabel = "T [K]" if inState in statesWithTemperature else "r"

# === Load FullLogNeg ===
for i in range(1, numTemperatures + 1):
    files = list(baseDir.glob(f"{TypeOfData.FullLogNeg.value}_{simulationName}_date_*_plotNumber_{i}.txt"))
    filePath = [f for f in files if isDateInRange(f)][0]
    arr = np.loadtxt(filePath)
    paramValues.append(arr[0])
    data[TypeOfData.FullLogNeg.value].append(arr[1:])

# === Load other types ===
for key in [TypeOfData.OddVSEven.value, TypeOfData.SameParity.value]:
    for i in range(1, numTemperatures + 1):
        files = list(baseDir.glob(f"{key}_{simulationName}_date_*_plotNumber_{i}.txt"))
        filePath = [f for f in files if isDateInRange(f)][0]
        arr = np.loadtxt(filePath)
        data[key].append(arr[1:])

# === Main loop per parameter value ===
for i in range(numTemperatures):
    modes = np.arange(1, len(data[TypeOfData.FullLogNeg.value][i]) + 1)
    residual = data[TypeOfData.FullLogNeg.value][i] - (
        data[TypeOfData.OddVSEven.value][i] + data[TypeOfData.SameParity.value][i]
    )

    # === Detect monogamy violations ===
    violated_modes = [(modes[j], abs(residual[j])) for j in range(len(modes)) if residual[j] < 0]

    # === Print violations ===
    if violated_modes:
        print(f"\n--- Monogamy violated for {paramLabel} = {paramValues[i]:.2f} ---")
        for mode_idx, res in violated_modes:
            print(f"Mode {mode_idx}: Abs Residual = {res:.4e}")
    else:
        print(f"No monogamy violation for {paramLabel} = {paramValues[i]:.2f}")

    # === Plot ===
    plt.figure(figsize=(10, 6))
    for key in logNegTypes:
        plt.plot(modes, data[key][i], label=labels[key], color=colors[key])
    plt.plot(modes, residual, label="Residual: Full - (Odd + Same)", linestyle='--', color='black')

    # === Highlight violated modes with yellow circles ===
    if violated_modes:
        v_modes, v_vals = zip(*violated_modes)
        plt.scatter(v_modes, v_vals, color='yellow', edgecolor='red', label="Monogamy Violated", zorder=5, s=100, marker='o')

    # === Labels and titles ===
    paramVal = paramValues[i]
    paramStr = sanitize(f"{paramVal:.2f}")
    extraLabel = ""
    extraFilename = ""

    if inState == InitialState.ThermalFixedOneModeSqueezing.value:
        extraLabel = f", r = {rManual:.2f}"
        extraFilename = f"_r_{sanitize(rManual)}"
    elif inState in {
        InitialState.OneModeSqueezedFixedTemp.value,
        InitialState.TwoModeSqueezedFixedTemp.value
    }:
        extraLabel = f", T = {tManual:.3f} K"
        extraFilename = f"_T_{sanitize(tManual)}K"

    # === Save residuals and violations ===
    residualPath = plotsDir / f"residual_{paramLabel.replace(' ', '')}_{paramStr}{extraFilename}_{simulationName}.txt"
    np.savetxt(residualPath, np.column_stack((modes, residual)), header="ModeIndex Residual")

    if violated_modes:
        violationsPath = plotsDir / f"monogamyViolations_{paramLabel.replace(' ', '')}_{paramStr}{extraFilename}_{simulationName}.txt"
        np.savetxt(violationsPath, np.array(violated_modes), header="ModeIndex AbsResidual")

    # === Final plot config ===
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"LogNeg comparison at {paramLabel} = {paramVal:.2f}{extraLabel}")
    plt.xlabel("Mode index (log scale)")
    plt.ylabel("Logarithmic Negativity (log scale)")
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()

    savePath = plotsDir / f"loglog_logneg_comparison_{paramLabel.replace(' ', '')}_{paramStr}{extraFilename}_{simulationName}.pdf"
    plt.savefig(savePath)
    plt.close()

