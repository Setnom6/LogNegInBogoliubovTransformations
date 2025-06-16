from scipy.linalg import null_space

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
