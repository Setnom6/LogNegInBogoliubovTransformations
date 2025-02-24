import json
import os
import sys
import time
from LogNegManager import InitialState, LogNegManager, TypeOfData

# Function to ensure a directory exists, and create it if it doesn't
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get the configuration file from the command line arguments
if len(sys.argv) < 2:
    print("Usage: python simulate.py <config.json>")
    sys.exit(1)

config_file_path = sys.argv[1]

# Load configuration from the provided config file
with open(config_file_path) as config_file:
    config = json.load(config_file)

# Set default values for the inputs
MODES = config.get("MODES", 128)
instant = config.get("instant", 10)
inStateType = InitialState[config.get("inStateType", "Vacuum")]
arrayParameters = config.get("arrayParameters", [0.0])
temperature = config.get("temperature", None)
squeezingIntensity = config.get("squeezingIntensity", None)
dataDirectory = config.get("dataDirectory", "./sims-128/")
plotsDirectory = config.get("plotsDirectory", "./plots/128-1plt-plots/")
plotsDataDirectory = config.get("plotsDataDirectory", "./plotsData/128-1plt-data/")
specialModes = config.get("specialModes", [1, 2, 3, 4, 5])
listOfWantedComputations = [TypeOfData[comp] for comp in config.get("listOfWantedComputations", ["FullLogNeg"])]
tryToLoad = config.get("tryToLoad", True)
saveFig = config.get("saveFig", True)

# Ensure that the directories exist
ensure_directory_exists(dataDirectory)
ensure_directory_exists(plotsDirectory)
ensure_directory_exists(plotsDataDirectory)

# Start the timer
start_time = time.time()

# Initialize the simulation
simulation = LogNegManager(dataDirectory, inStateType, MODES, instant, arrayParameters,
                           temperature=temperature, squeezingIntensity=squeezingIntensity)

# Perform the transformation to the initial state
simulation.performTransformation()

# Perform the computations
results = simulation.performComputations(listOfWantedComputations, plotsDataDirectory, tryToLoad, specialModes)

# Stop the timer
end_time = time.time()
elapsed_time = end_time - start_time

# Generate and save the plots
simulation.generatePlots(results, plotsDirectory, plotsDataDirectory, specialModes, listOfWantedComputations, saveFig=saveFig)

# Print the elapsed time
print(f"Total computation time: {elapsed_time:.2f} seconds")
