import os
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.linalg import null_space

from LogNegManager import InitialState, LogNegManager
from qgt import *


def extractHawkingPartner(bogoliubovTransformation, modeIndex, atol=1e-8):
    """
    Extracts the basis transformation that maps the original OUT basis to a new one
    where mode 0 is the Hawking mode, mode 1 is the partner, and the rest are reordered.

    Parameters:
        bogoliubovTransformation: 2N x 2N matrix mapping {a_in, a_in^†} → {a_out, a_out^†}
        modeIndex: index of the Hawking mode in the original basis (between 0 and N-1)
        atol: tolerance for checking commutators

    Returns:
        newBogoliubovTransformation: Bogoliubov transformation in the new ordered basis
        HPExpressedInOUTBasis: 2N x 2N transformation from the original OUT basis → new basis
    """
    alphaVector, betaVector = getAlphaBetaVectors(bogoliubovTransformation, modeIndex)

    nParallel, alpha, nPerp, betaParallelConj, betaPerpConj, remainingOrthonormals = computeBetaParallelAndPerp(
        alphaVector, betaVector, atol=atol)

    gp, gpp, dp, dpp = computePartnerMode(alpha, betaParallelConj, betaPerpConj, atol=atol)

    fromInBasisToParallelPerp = buildFullInToParallelPerpMatrix(
        nParallel, nPerp, remainingOrthonormals, bogoliubovTransformation.shape[0] // 2)
    fromParallelPerpToHawkingPartner = buildFullParallelPerpToHawkingPartner(
        alpha, betaParallelConj, betaPerpConj, gp, gpp, dp, dpp, len(remainingOrthonormals))

    newBogoliubovTransformation = fromParallelPerpToHawkingPartner @ fromInBasisToParallelPerp
    HPExpressedInOUTBasis = newBogoliubovTransformation @ np.linalg.inv(bogoliubovTransformation)

    if not Is_Sympletic(newBogoliubovTransformation, 1, atol=atol):
        raise ValueError("New Bogoliubov transformation is not symplectic")

    # Create the basis vector representing the selected mode in the OUT basis
    projectorOnSelectedMode = np.zeros(bogoliubovTransformation.shape[0])
    projectorOnSelectedMode[2 * modeIndex] = 1

    # Extract the row corresponding to the Hawking mode in the OUT basis
    hawkingRow = HPExpressedInOUTBasis[0]
    hawkingDifference = hawkingRow - projectorOnSelectedMode
    if np.any(np.abs(hawkingDifference) > atol):
        raise ValueError(
            f"The Hawking mode does not match the selected OUT mode (index {modeIndex})."
        )

    # Extract the row corresponding to the Partner mode
    partnerRow = HPExpressedInOUTBasis[2]
    partnerContributionInHawkingMode = partnerRow[2 * modeIndex]
    if np.abs(partnerContributionInHawkingMode) > atol:
        raise ValueError(
            f"The Partner mode has a non-zero contribution in the selected OUT mode (index {modeIndex})."
        )

    return newBogoliubovTransformation, HPExpressedInOUTBasis


def getAlphaBetaVectors(S, modeIndex):
    """
    Extracts the alpha and beta vectors from the matrix S for a given mode.

    Parameters:
        S: numpy matrix of dimension (2n, 2n), can be complex
        modeIndex: integer (index of the mode, starting from 0)

    Returns:
        alphaVector: elements at even positions of row 2*modeIndex, conjugated
        betaVector: elements at odd positions of row 2*modeIndex, as they are
    """
    row = S[2 * modeIndex]
    alphaVector = np.conj(row[::2])  # elements 0, 2, 4, ... conjugated
    betaVector = np.conj(-row[1::2])  # elements 1, 3, 5, ... not conjugated
    return alphaVector, betaVector


def computeBetaParallelAndPerp(alphaVec, betaVec, atol=1e-8):
    """
    Given an alphaVec and betaVec, returns:
        - nParallel: unit vector (with phase) in the direction of alphaVec
        - alpha: COMPLEX scalar component of alphaVec on nParallel
        - nPerp: unit vector orthogonal to nParallel, in the direction of betaPerp
        - betaParallel: COMPLEX scalar component of betaVec on nParallel
        - betaPerp: COMPLEX scalar component of betaVec on nPerp
        - remainingOrthonormals: vectors orthogonal to nParallel and nPerp
    """
    dim = len(alphaVec)

    alphaNorm = np.linalg.norm(alphaVec)
    if alphaNorm < atol:
        raise ValueError("The alpha vector cannot be null.")

    # Parallel vector (Hawking mode)
    nParallel = alphaVec / alphaNorm
    alpha = np.vdot(nParallel, alphaVec)
    betaConj = np.conj(betaVec)
    betaParallel_conj = np.vdot(nParallel, betaConj)

    # Perpendicular vector
    betaResidual = betaConj - betaParallel_conj * nParallel
    betaPerpNorm = np.linalg.norm(betaResidual)

    if betaPerpNorm > atol:
        nPerp = betaResidual / betaPerpNorm
        betaPerp_conj = np.vdot(nPerp, betaConj)
        basis = np.column_stack([nParallel, nPerp])
    else:
        nPerp = np.zeros_like(betaConj)
        betaPerp_conj = 0.0
        basis = np.atleast_2d(nParallel).T

    # Check commutator
    commutator = abs(alpha) ** 2 - abs(betaParallel_conj) ** 2 - abs(betaPerp_conj) ** 2
    if abs(commutator - 1) > 1e-6:
        raise ValueError(f"Commutator is not 1: {commutator}")

    # Orthonormal basis of the orthogonal complement
    nullspace = null_space(basis.conj().T)  # columns: orthonormal vectors

    # Dimension check
    expected = dim - basis.shape[1]
    if nullspace.shape[1] != expected:
        raise ValueError(f"The orthonormal basis has {nullspace.shape[1]} vectors, but {expected} were expected")

    # Convert to list
    remainingOrthonormals = [nullspace[:, i] for i in range(nullspace.shape[1])]

    return nParallel, alpha, nPerp, betaParallel_conj, betaPerp_conj, remainingOrthonormals


def computePartnerMode(alpha, betaParallelConj, betaPerpConj, normalize=True, atol=1e-8):
    """
    Given a mode a_H = alpha * a_|| + betaParallel * a_||^† + betaPerp * a_perp^†,
    returns the coefficients of the partner mode:

    a_p = gammaParallel^* a_|| + gammaPerp^* a_perp +
          deltaParallel a_||^† + deltaPerp a_perp^†

    All coefficients are complex scalars.
    """
    betaParallel = np.conj(betaParallelConj)
    betaPerp = np.conj(betaPerpConj)

    if np.abs(betaParallel) < atol:
        # Special case: betaParallel = 0
        gammaParallel = 0.0 + 1j * 0
        gammaPerp = alpha
        deltaParallel = betaPerp
        deltaPerp = 0.0 + 1j * 0

    elif np.abs(betaPerp) < atol:
        # Special case: betaPerp = 0 (no partner needed, it is itself)
        gammaParallel = alpha
        gammaPerp = 0
        deltaParallel = 0
        deltaPerp = 0

        print("No partner needed. Some calculations may fail from now on")
    else:
        # Option B1 from paper 1503.06109 modified

        # Set deltaPerp = 0
        deltaPerp = 0.0 + 1j * 0

        # Define factors
        A = betaParallelConj / np.conj(alpha)
        B = (alpha - A * betaParallel) / betaPerp

        # Solve the equation:
        # |A * d|^2 + |B * d|^2 - |d|^2 = 1
        # => |d|^2 * (|A|^2 + |B|^2 - 1) = 1
        normFactor = np.abs(A) ** 2 + np.abs(B) ** 2 - 1

        if np.abs(normFactor) < atol:
            raise ValueError("Norm condition leads to divergence (denominator ~ 0)")

        # Absolute value of deltaParallel is easily obtained
        absDeltaParallel = np.sqrt(1 / normFactor)
        deltaParallel = absDeltaParallel  # We choose deltaParallel to be real by choosing the phase

        # Now calculate the gamma coefficients
        gammaParallel = A * deltaParallel
        gammaPerp = B * deltaParallel

    conmutatorPartner = np.abs(gammaParallel) ** 2 + np.abs(gammaPerp) ** 2 - np.abs(deltaParallel) ** 2 - np.abs(
        deltaPerp) ** 2

    hawkingPartnerConmutator = np.conj(gammaParallel) * alpha - betaParallel * np.conj(
        deltaParallel) - betaPerp * np.conj(deltaPerp)

    partnerHawkingConmutator = np.conj(gammaParallel) * betaParallelConj + np.conj(gammaPerp) * betaPerpConj - np.conj(
        alpha) * np.conj(deltaParallel)

    if abs(conmutatorPartner - 1.0) > 1e-4:
        raise ValueError("Partner commutation relation fails to be fulfilled")

    if abs(hawkingPartnerConmutator) > 1e-4 or abs(partnerHawkingConmutator) > 1e-4:
        raise ValueError("Hawking partner commutation fails to be fulfilled")

    return gammaParallel, gammaPerp, deltaParallel, deltaPerp


def buildFullParallelPerpToHawkingPartner(alpha, betaParallelConj, betaPerpConj, gp, gpp, dp, dpp, numPerpendiculars):
    """
    Construye la matriz de cambio de base de la forma:
    {a_||, a_||^†, a_perp, a_perp^†, a_perp1, a_perp1^†, ...} →
    {a_H, a_H^†, a_P, a_P^†, a_perp1, a_perp1^†, ...}

    El número total de modos N = 2 (|| y perp) + numPerpendiculars

    Retorna:
        fromParallelPerpToHawkingPartner: matriz (2N x 2N)
    """
    totalRows = 2 * (2 + numPerpendiculars)
    M = np.eye(totalRows, dtype=complex)

    betaParallel = np.conj(betaParallelConj)
    betaPerp = np.conj(betaPerpConj)

    # Reemplazar el bloque superior izquierdo 4x4 con la transformación Hawking+Partner
    M[0:4, 0:4] = np.array([
        [np.conj(alpha), -betaParallelConj, 0, -betaPerpConj],
        [-betaParallel, alpha, -betaPerp, 0],
        [np.conj(gp), -np.conj(dp), np.conj(gpp), -np.conj(dpp)],
        [-dp, gp, -dpp, gpp]
    ])

    return M


def buildFullInToParallelPerpMatrix(nParallel, nPerp, remainingOrthonormals, totalModes):
    """
    Builds the unitary matrix that maps the original IN basis to the basis
    {a_||, a_perp, a_perp1, ...}, along with their conjugates.

    Parameters:
        nParallel: complex unit vector (dim = totalModes)
        nPerp: complex unit vector orthogonal to nParallelConj (or null vector)
        remainingOrthonormals: list of unit vectors orthogonal to the rest
        totalModes: total number of modes (N)

    Returns:
        U: (2N x 2N) matrix for basis transformation of annihilation and creation operators
    """
    basis = [nParallel]
    if np.linalg.norm(nPerp) > 0:
        basis.append(nPerp)
    basis += remainingOrthonormals

    if len(basis) != totalModes:
        raise ValueError(f"The orthonormal basis has {len(basis)} vectors, but {totalModes} were expected")

    A = np.column_stack(basis)  # (N x N) matrix that transforms the annihilation basis

    # Build the transformation for {a, a†}
    U = np.zeros((2 * totalModes, 2 * totalModes), dtype=complex)
    U[::2, ::2] = np.conj(A.T)  # a' = sum_j A*_ij a_j
    U[1::2, 1::2] = A.T  # a'† = sum_j A^T_ij a_j†

    return U


def get_symplectic_from_covariance(V):
    """
    Computes the symplectic matrix S from the covariance matrix V, such that V = S S^T.

    Parameters:
        V: covariance matrix

    Returns:
        S: symplectic matrix
    """
    S = sqrtm(V)
    return S


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
