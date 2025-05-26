# -*- coding: utf-8 -*-
"""
QuGIT_v2 - Quantum Gaussian Information Toolbox Modified to hbar=1

Author: Jose Manuel Montes, based on Igor Brandão QuGIT

In this toolbox the following conventions are used:

hbar = 1
q=1/Sqrt(2)*(a+a^{dagger})      (quadrature operators as a function of Fock's space operators)
p=i/Sqrt(2)*(a^{dagger}-a)

Then [q,p]=i*Omega              where Omega is the symplectic form

x=(q1,p1,q2,p2,...,qn,pn)       vector of quadratures

R^i} = <x^i>                    mean vector elements

V^ij = <{x^i-R^i,x^j-R^j}>      covariance matrix without factor 1/2 and anticonmutator {A,B}=AB+BA

Those conventions lead to:

Heisenberg relation: V+i*Omega >= 0
Vacuum cov matrix: V_vac = Identity_matrix
"""

import warnings

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import block_diag
from scipy.linalg import sqrtm


################################################################################

class Gaussian_state:  # Class definning a multimode gaussian state
    """Class simulation of a multimode gaussian state
    
    ATTRIBUTES:
        self.R       - Mean quadratures vector
        self.V       - Covariance matrix
        self.Omega   - Symplectic form matrix
        self.N_modes - Number of modes
    """

    # Constructor and its auxiliar functions    
    def __init__(self, *args):
        """
        The user can explicitly pass the first two moments of a multimode gaussian state
        or pass a name-value pair argument to choose a single mode gaussian state.
        
        PARAMETERS:
            R0, V0 - mean quadratures vector and covariance matrix of a gaussian state (ndarrays)
            
        NAME-VALUE PAIR ARGUMENTS:
            "vacuum" , number of modes                                       			- generates vacuum   state (string, int)
            "thermal" , number of modes, mean number of thermal quanta (occupation number)      - generates thermal  state (string, int, float)
            "coherent", number of modes, complex amplitude                   			- generates coherent state (string, int, complex)
            "squeezed", number of modes, squeezing parameter                 			- generates squeezed state (string, int, float)
        """

        if (len(args) == 0):  # Default constructor (single mode vacuum state)
            self.R = np.array([[0], [0]])  # Save mean quadratres   in a class attribute
            self.V = np.identity(2)  # Save covariance matrix in a class attribute
            self.N_modes = 1

        elif (isinstance(args[0], str)):  # If the user called for an elementary gaussian state
            self.decide_which_state(args)  # Call the proper method to decipher which state the user wants

        elif (isinstance(args[0], np.ndarray) and isinstance(args[1],
                                                             np.ndarray)):  # If the user gave the desired mean quadratures values and covariance matrix
            R0 = args[0]
            V0 = args[1]

            R_is_real = np.all(np.isreal(R0))
            R_is_vector = np.squeeze(R0).ndim == 1

            V_is_matrix = np.squeeze(V0).ndim == 2
            V_is_square = V0.shape[0] == V0.shape[1]

            R_and_V_match = len(R0) == len(V0)

            assert R_is_real and R_is_vector and V_is_matrix and R_and_V_match and V_is_square, "Unexpected first moments when creating gaussian state!"  # Make sure they are a vector and a matrix with same length

            self.R = np.vstack(R0);  # Save mean quadratres   in a class attribute (vstack to ensure column vector)
            self.V = V0;  # Save covariance matrix in a class attribute
            self.N_modes = int(len(R0) / 2);  # Save the number of modes of the multimode state in a class attribute

        else:
            raise ValueError(
                'Unexpected arguments when creating gaussian state!')  # If input arguments do not make sense, call out the user

        omega = np.array([[0, 1], [-1, 0]]);  # Auxiliar variable
        self.Omega = np.kron(np.eye(self.N_modes, dtype=int),
                             omega)  # Save the symplectic form matrix in a class attribute, the Kronecker product with the identity matrix is equals the direct sum of matrices which are equal

    def decide_which_state(self, varargin):
        # If the user provided a name-pair argument to the constructor,
        # this function reads these arguments and creates the first moments of the gaussian state

        assert isinstance(varargin[1], int) and varargin[
            1] > 0, "Number of modes non (well) specified"  # Checks if the number of modes is said and is an int
        modes = varargin[1]
        type_state = varargin[0];  # Name of expected type of gaussian state
        self.N_modes = modes

        if (str(type_state) == "vacuum"):  # If it is a vacuum state
            self.R = np.zeros((2 * modes, 1))  # Save mean quadratres   in a class attribute
            self.V = np.identity(2 * modes)  # Save covariance matrix in a class attribute
            return  # End function

            # Make sure there is an extra parameters that is a number
        assert len(varargin) > 2, "Absent amplitude for non-vacuum elementary gaussian state"
        assert isinstance(varargin[2],
                          (int, float, complex)), "Invalid amplitude for non-vacuum elementary gaussian state"

        if (str(type_state) == "thermal"):  # If it is a thermal state
            nbar = varargin[2]
            assert nbar >= 0, "Imaginary or negative occupation number for thermal state"  # Make sure its occuption number is a non-negative number
            self.R = np.zeros((2 * modes, 1))
            aux_v = np.diag([2.0 * nbar + 1, 2.0 * nbar + 1])  # With a variance aux variable we create its variance
            self.V = np.kron(np.eye(modes, dtype=int), aux_v)  # Create its first moments
            return

        elif (str(type_state) == "coherent"):  # If it is a coherent state
            alpha = varargin[2]
            r_aux = np.tile([2 * alpha.real, 2 * alpha.imag], modes)  # Create mean vector
            self.R = np.vstack(r_aux)
            self.V = np.identity(2 * modes);  # Create its covariance matrix
            return

        elif (str(type_state) == "squeezed"):  # If it is a squeezed state
            r = varargin[2];  # Make sure its squeezing parameter is a real number
            assert np.isreal(r), "Unsupported imaginary amplitude for squeezed state"
            self.R = np.zeros((2 * modes, 1))
            aux_v = np.diag([np.exp(-2 * r), np.exp(+2 * r)])  # With a variance aux variable we create its variance
            self.V = np.kron(np.eye(modes, dtype=int), aux_v);  # Create its first moments
            return

        else:
            self.N_modes = []
            raise ValueError(
                "Unrecognized gaussian state name, please check for typos or explicitely pass its first moments as arguments")

    def check_uncertainty_relation(self):
        """
        Check if the generated covariance matrix indeed satisfies the uncertainty principle (debbugging)
        """

        V_check = self.V + 1j * self.Omega
        eigvalue, eigvector = np.linalg.eig(V_check)

        # assert all(eigvalue>=0), "CM does not satisfy uncertainty relation!"   This is the theoretical condition, but numerical errors can make it fail

        eigvalue = np.array([i for i in eigvalue if np.abs(i) >= 1e-8])  # We remove the numerical errors
        assert np.all(eigvalue >= 0), "CM does not satisfy uncertainty relation!"
        return V_check

    def __str__(self):
        return str(self.N_modes) + "-mode gaussian state with mean quadrature vector R =\n" + str(
            self.R) + "\nand covariance matrix V =\n" + str(self.V)

    def copy(self):
        """Create identical copy, and doesn't modify the original"""

        return Gaussian_state(self.R, self.V)

    # Construct another state, from this base gaussian_state
    def tensor_product_append(self, rho_list):
        """ Given a list of gaussian states, 
        # calculates the tensor product of the base state and the states in the array and changes the original
        # 
        # PARAMETERS:
        #    rho_array - array of gaussian_state (multimodes)
        #
         CALCULATES:
            rho - multimode gaussian_state with all of the input states
        """

        R_final = self.R;  # First moments of resulting state is the same of rho_A
        V_final = self.V;  # First block diagonal entry is the CM of rho_A

        for rho in rho_list:  # Loop through each state that is to appended
            R_final = np.vstack((R_final, rho.R))  # Create its first moments
            V_final = block_diag(V_final, rho.V)

        temp = Gaussian_state(R_final, V_final);  # Generate the gaussian state with these moments

        self.R = temp.R  # Copy its attributes into the original instance
        self.V = temp.V
        self.Omega = temp.Omega
        self.N_modes = temp.N_modes

    def partial_trace(self, indexes):
        """
        Partial trace over specific single modes of the complete gaussian state
        
        PARAMETERS:
           indexes - the modes the user wants to trace out (as in the mathematical notation) 
        
        CALCULATES:
           rho_A - gaussian_state with all of the input state, except of the modes specified in 'indexes'
        """

        N_A = int(len(self.R) - 2 * len(indexes));  # Twice the number of modes in resulting state
        assert N_A >= 0, "Partial trace over more states than there exist in gaussian state"
        assert max(indexes) < self.N_modes, "Cannot trace out modes that do not exist"

        modes = np.arange(self.N_modes)  # These lines create an array of the modes we want to keep
        entries = np.isin(modes, indexes)
        entries = [not elem for elem in entries]
        modes = modes[entries]

        R0 = np.zeros((N_A, 1))
        V0 = np.zeros((N_A, N_A))

        for i in range(len(modes)):
            m = modes[i]
            R0[(2 * i):(2 * i + 2)] = self.R[(2 * m):(2 * m + 2)]

            for j in range(len(modes)):
                n = modes[j]
                V0[(2 * i):(2 * i + 2), (2 * j):(2 * j + 2)] = self.V[(2 * m):(2 * m + 2), (2 * n):(2 * n + 2)]

        temp = Gaussian_state(R0, V0);  # Generate the gaussian state with these moments

        self.R = temp.R  # Copy its attributes into the original instance
        self.V = temp.V
        self.Omega = temp.Omega
        self.N_modes = temp.N_modes

    def only_modes(self, indexes):
        """
        Partial trace over all modes except the ones in indexes of the complete gaussian state

         PARAMETERS:
            indexes - the modes the user wants to retrieve from the multimode gaussian state

         CALCULATES:
            rho - gaussian_state with all of the specified modes
        """
        N_A = len(indexes);  # Number of modes in resulting state
        assert N_A > 0 and N_A <= self.N_modes, "Partial trace over more states than exists in gaussian state"

        R0 = np.zeros((2 * N_A, 1), dtype=np.complex_)
        V0 = np.zeros((2 * N_A, 2 * N_A), dtype=np.complex_)

        for i in range(len(indexes)):
            m = indexes[i]
            R0[(2 * i):(2 * i + 2)] = self.R[(2 * m):(2 * m + 2)]

            for j in range(len(indexes)):
                n = indexes[j]
                V0[(2 * i):(2 * i + 2), (2 * j):(2 * j + 2)] = self.V[(2 * m):(2 * m + 2), (2 * n):(2 * n + 2)]

        temp = Gaussian_state(R0, V0);  # Generate the gaussian state with these moments

        self.R = temp.R  # Copy its attributes into the original instance
        self.V = temp.V
        self.Omega = temp.Omega
        self.N_modes = temp.N_modes

        # Properties of the gaussian state

    def symplectic_eigenvalues(self):
        """
        Calculates the sympletic eigenvalues of a covariance matrix V with symplectic form Omega
        
        Finds the absolute values of the eigenvalues of i\Omega V and removes repeated entries
        
        CALCULATES:
            lambda - array with symplectic eigenvalues
        """
        H = 1j * np.matmul(self.Omega, self.V);  # Auxiliar matrix
        lambda_0, v_0 = np.linalg.eig(H)
        lambda_0 = np.abs(lambda_0);  # Absolute value of the eigenvalues of the auxiliar matrix

        lambda_s = np.zeros((self.N_modes, 1));  # Variable to store the symplectic eigenvalues
        for i in range(self.N_modes):  # Loop over the non-repeated entries of lambda_0
            lambda_s[i] = lambda_0[0]  # Get the first value on the repeated array
            lambda_0 = np.delete(lambda_0, 0)  # Delete it

            idx = np.argmin(np.abs(lambda_0 - lambda_s[i]))  # Find the next closest value on the array (repeated entry)
            lambda_0 = np.delete(lambda_0, idx)  # Delete it too

        lambda_s = np.vstack(np.sort(lambda_s, axis=None))  # Sort all of them to get the less value the first
        return lambda_s  # the structure is [[v1],[v2],[v3],...]

    def check_uncertainty_relation_eigenvalues(self):
        """
        Another function to check if the covariance matrix is a bona fide covariance matrix:

        It calculates the symplectic eigenvalues and checks if all of them are more than 1
        """

        eigs = self.symplectic_eigenvalues()
        tol = 1e-7  # Tolerance for computation errors
        assert eigs[0][0] >= 1 - tol, "The covariance matrix is not a bona fide cov matrix"
        return True

    def purity(self):
        """
        Purity of a gaussian state (pure states have unitary purity)

         CALCULATES:
             p - purity
        """

        return 1 / np.prod(self.symplectic_eigenvalues())

    def partial_transpose(self, *args):
        """
        Calculation of the partial transpose covariance of the gaussian state and store it in a new gaussian state
        subsystem B is the one which is transposed

        PARAMETERS:
            subsA: list of modes to be considered the subsystem A
            subsB: list of modes to be considered the subsystem B
            they should be two separate arrays
            if only one array is introduced, it is considered the subsA

        CALCULATES:
            PTstate - Gaussian state with same mean vector as input but with partial transpose covariance matrix
        """

        modes = self.N_modes
        if (modes == 2):  # If the full system is only comprised of two modes
            subsA = [0]
            subsB = [1]
        elif (len(args) == 1 and modes > 2):  # Only ons subsystem specified
            assert all(0 <= x < self.N_modes for x in args[0]), "subsA should be composed of valid modes in the state"
            subsA = args[0]
            subsB = np.arange(self.N_modes)  # These lines create an array of the modes which will be subsB
            entries = np.isin(subsB, subsA)
            entries = [not elem for elem in entries]
            subsB = subsB[entries]

        elif (len(args) > 1 and modes > 2):
            subsA = args[0]
            subsB = args[1]
            assert not np.any(np.intersect1d(subsA, subsB)), "Each mode should be only in one subsystem"
            assert len(subsA) > 0 and len(subsB) > 0 and len(subsA) + len(
                subsB) == modes, "Number of modes out of range, all modes should be considered"
            assert np.all(len(subsA) <= modes - 1) and np.all(len(subsA) >= 0) and np.all(
                len(subsB) <= modes - 1) and np.all(len(subsB) >= 0), "Indices are out of range"

        else:
            raise TypeError(
                'Unable to decide which bipartite entanglement to infer, please pass the indexes to the desired bipartition')

        # We have to create the matrix T of 2N Identity matrices and M pauli matrices
        # But we have to take into account that the subsA modes are the
        # ones that are transposed, so we have to order the matrices Z and Identities
        # in the correct order

        T = np.eye(2 * modes)
        for i in range(modes):
            if i in subsA:
                T[2 * i + 1][2 * i + 1] = -1  # We change the sign of the momentum quadrature of the modes in subsA
        """Zpauli=np.array([[1, 0], [0, -1]])                                      #We create the matrix T of 2N Identity matrices and M pauli matrices
        T=block_diag(np.eye(2*len(subsA)),Zpauli)
        for i in range(len(subsB)-1):
            T=block_diag(T,Zpauli)"""
        Vpt = np.matmul(np.matmul(T, self.V), T)  # Create the partial transpose covariance matrix
        return Gaussian_state(np.zeros(2 * (len(subsA) + len(subsB))), Vpt)

    def von_Neumann_Entropy(self):
        """
        Calculation of the von Neumann entropy for a multipartite gaussian system
       
        CALCULATES:
             Entropy - von Neumann entropy of the multimode state
        """

        nu = self.symplectic_eigenvalues();  # Calculates the sympletic eigenvalues of a covariance matrix V

        # 0*log(0) is NaN, but in the limit that x->0 : x*log(x) -> 0
        # nu[np.abs(nu - 1) < 1e-15] = nu[np.abs(nu - 1) < 1e-15] + 1e-15;                                 # Doubles uses a 15 digits precision, I'm adding a noise at the limit of the numerical precision
        nu[np.abs(nu - 1) < 1e-15] = 1 + 1e-15

        nu_plus = (nu + 1) / 2.0;  # Temporary variables
        # nu_minus = (nu - 1)/2.0;
        nu_minus = np.abs((nu - 1) / 2.0)
        g_nu = np.multiply(nu_plus, np.log2(nu_plus)) - np.multiply(nu_minus, np.log2(nu_minus))

        Entropy = np.sum(g_nu);  # Calculate the entropy
        return Entropy

    def occupation_number(self):
        """
        Occupation number for a each single mode within the multipartite gaussian state (array)
        
        CALCULATES:
            nbar - array with the occupation number for each single mode of the multipartite gaussian state
        """

        Variances = np.diag(self.V);  # From the current CM, take the variances
        Variances = np.vstack(Variances)

        mean_x = self.R[::2];  # Odd  entries are the mean values of the position
        mean_p = self.R[1::2];  # Even entries are the mean values of the momentum

        Var_x = Variances[::2];  # Odd  entries are position variances
        Var_p = Variances[1::2];  # Even entries are momentum variances

        nbar = 0.25 * (
                Var_x + mean_x ** 2 + Var_p + mean_p ** 2) - 0.5;  # Calculate occupantion numbers at current time
        return nbar

    def number_operator_moments(self):
        """
        Calculates means vector and covariance matrix of photon numbers for each mode of the gaussian state
        
        CALCULATES:
            m - mean values of number operator in arranged in a vector (Nx1 numpy.ndarray)
            K - covariance matrix of the number operator               (NxN numpy.ndarray)
           
        REFERENCE:
            Phys. Rev. A 99, 023817 (2019)
            Many thanks to Daniel Tandeitnik for the base code for this method!
        """
        q = self.R[::2]  # Mean values of position quadratures (even entries of self.R)
        p = self.R[1::2]  # Mean values of momentum quadratures (odd  entries of self.R)

        alpha = (q + 1j * p) / np.sqrt(2)  # Mean values of annihilation operators
        alpha_c = (q - 1j * p) / np.sqrt(2)  # Mean values of creation     operators

        V_1 = self.V[0::2, 0::2] / 2.0  # Auxiliar matrix
        V_2 = self.V[0::2, 1::2] / 2.0  # Auxiliar matrix
        V_3 = self.V[1::2, 1::2] / 2.0  # Auxiliar matrix

        A = (V_1 + V_3 + 1j * (np.transpose(V_2) - V_2)) / 2.0  # Auxiliar matrix
        B = (V_1 - V_3 + 1j * (np.transpose(V_2) + V_2)) / 2.0  # Auxiliar matrix

        temp = np.multiply(np.matmul(alpha_c, alpha.transpose()), A) + np.multiply(
            np.matmul(alpha_c, alpha_c.transpose()), B)  # Yup, you guessed it, another auxiliar matrix

        m = np.real(np.reshape(np.diag(A), (self.N_modes, 1)) + np.multiply(alpha,
                                                                            alpha_c) - 0.5)  # Mean values of number operator (occupation numbers)

        K = np.real(np.multiply(A, A.conjugate()) + np.multiply(B, B.conjugate()) - 0.25 * np.eye(
            self.N_modes) + 2.0 * temp.real)  # Covariance matrix for the number operator

        return m, K

    def logarithmic_negativity(self, *args):
        """
        Calculation of the logarithmic negativity for a bipartite system
       
        PARAMETERS:
           subsA - array with indices for the subsistem A (subsA will have len N)
           subsB -array with indices for the subsystem B (subsB will have len M)
           If the system is already bipartite, this parameter is optional !
       
        CALCULATES:
           LN - logarithmic negativity for the bipartition / bipartite states
        """

        temp = self.N_modes
        if (temp == 2):  # If the full system is only comprised of two modes
            auxGaussian = self.copy()  # Take its full covariance matrix
            subsA = [0]
            subsB = [1]
        elif (len(args) == 1 and temp > 2):  # Only ons subsystem specified
            assert all(0 <= x < self.N_modes for x in args[0]), "subsA should be composed of valid modes in the state"
            subsA = args[0]
            subsB = np.arange(self.N_modes)  # These lines create an array of the modes which will be subsB
            entries = np.isin(subsB, subsA)
            entries = [not elem for elem in entries]
            subsB = subsB[entries]
            auxGaussian = self.copy()
            auxGaussian.only_modes(subsA + subsB)  # Otherwise, get only the modes for each subsystem
            subsA = [i for i in range(0, len(subsA))]  # rewrite the indexes for the new gaussian state
            subsB = [i for i in
                     range(len(subsA), auxGaussian.N_modes)]  # rewrite the indexes for the new gaussian state

        elif (len(args) > 1 and temp > 2):
            subsA = args[0]
            subsB = args[1]

            assert not np.any(np.intersect1d(subsA, subsB)), "Each mode should be only in one subsystem"
            assert len(subsA) > 0 and len(subsB) > 0 and len(subsA) + len(subsB) <= temp, "Number of modes out of range"
            assert np.all(len(subsA) <= temp - 1) and np.all(len(subsA) >= 0) and np.all(
                len(subsB) <= temp - 1) and np.all(len(subsB) >= 0), "Indices are out of range"

            if len(subsA) > 1:
                warnings.warn(
                    "The subsystem A is not a single mode, LN cannot assure when the state is entangled unequivocally")

            auxGaussian = self.copy()
            auxGaussian.only_modes(subsA + subsB)  # Otherwise, get only the modes for each subsystem
            subsA = [i for i in range(0, len(subsA))]  # rewrite the indexes for the new gaussian state
            subsB = [i for i in
                     range(len(subsA), auxGaussian.N_modes)]  # rewrite the indexes for the new gaussian state
        else:
            raise TypeError(
                'Unable to decide which bipartite entanglement to infer, please pass the indexes to the desired bipartition')

        eigs = auxGaussian.partial_transpose(subsA,
                                             subsB).symplectic_eigenvalues()  # create the partial transpose and generate its eigenvalues
        LN = 0
        for i, value in enumerate(eigs):
            LN = LN + np.max([0, -np.log2(value[
                                              0])])  # Calculate the logarithmic negativity. value[0] is used because eigs has de shape [[vi],[v2],...]
        return LN

    def logarithmic_negativity_two_modes(self, *args):
        """
        Calculation of the logarithmic negativity for a bipartite system of 1 mode each
       
        PARAMETERS:
           indexes - array with indices for the bipartition to consider 
           If the system is already bipartite, this parameter is optional !
       
        CALCULATES:
           LN - logarithmic negativity for the bipartition / bipartite states
        """

        temp = self.N_modes
        if (temp == 2):  # If the full system is only comprised of two modes
            V0 = self.V  # Take its full covariance matrix
        elif (len(args) > 0 and temp > 2):
            indexes = args[0]

            assert len(indexes) == 2, "Can only calculate the logarithmic negativity for a bipartition!"

            bipartition = self.copy()
            bipartition.only_modes(indexes)  # Otherwise, get only the two mode specified by the user
            V0 = bipartition.V  # Take the full Covariance matrix of this subsystem
        else:
            raise TypeError(
                'Unable to decide which bipartite entanglement to infer, please pass the indexes to the desired bipartition')

        A = V0[0:2, 0:2]  # Make use of its submatrices
        B = V0[2:4, 2:4]
        C = V0[0:2, 2:4]

        sigma = np.linalg.det(A) + np.linalg.det(B) - 2.0 * np.linalg.det(C)  # Auxiliar variable

        ni = sigma / 2.0 - np.sqrt(sigma ** 2 - 4.0 * np.linalg.det(
            V0)) / 2.0;  # Square of the smallest of the symplectic eigenvalues of the partially transposed covariance matrix

        if (
                ni < 0.0):  # Manually perform a maximum to save computational time (calculation of a sqrt can take too much time and deal with residual numeric imaginary parts)
            LN = 0.0;
        else:
            ni = np.sqrt(
                ni.real);  # Smallest of the symplectic eigenvalues of the partially transposed covariance matrix

        LN = np.max([0, -np.log(ni)]);  # Calculate the logarithmic negativity at each time
        return LN

    def add_thermal_noise(self, nbar):
        """
        Add thermal noise to the Gaussian state using the transformation V' = S V S^T

        ARGUMENTS:
            nbar - the environmental number of quanta. Could be a np.ndarray vector or a float
        """
        if isinstance(nbar, np.ndarray):
            assert len(nbar.shape) == 1, "Occupation number should be a vector or a float"
            assert len(nbar) == self.N_modes, "If occupation number is a vector, it should have length equal to N_modes"

            # Construct the scaling matrix S = ⨁ sqrt(1+2*n_i) I_2
            S = np.diag(np.repeat(np.sqrt(1 + 2 * nbar), 2))

        else:
            assert (isinstance(nbar, float) or isinstance(nbar,
                                                          int)) and nbar >= 0, "Occupation number should be a non-negative float or int"

            # If nbar is a scalar, apply the same scaling to all modes
            S = np.sqrt(1 + 2 * nbar) * np.eye(2 * self.N_modes)

        # Apply the transformation using apply_unitary
        d = np.zeros_like(self.R)  # No displacement component
        self.apply_unitary(S, d)

    def fidelity(self, rho_2):
        """
        Calculates the fidelity between the two arbitrary gaussian states
        
        ARGUMENTS:
            rho_1, rho_2 - gaussian states to be compared through fidelity
         
        CALCULATES:
            F - fidelity between rho_1 and rho_2
        
        REFERENCE:
            Phys. Rev. Lett. 115, 260501.
       
        OBSERVATION:
        The user should note that non-normalized quadratures are expected;
        They are normalized to be in accordance with the notation of Phys. Rev. Lett. 115, 260501.
        """

        assert self.N_modes == rho_2.N_modes, "Impossible to calculate the fidelity between gaussian states of diferent sizes!"

        u_1 = self.R / np.sqrt(2.0);  # Normalize the mean value of the quadratures
        u_2 = rho_2.R / np.sqrt(2.0);

        V_1 = self.V / 2.0;  # Normalize the covariance matrices
        V_2 = rho_2.V / 2.0;

        OMEGA = self.Omega;
        OMEGA_T = np.transpose(OMEGA)

        delta_u = u_2 - u_1;  # A bunch of auxiliar variables
        delta_u_T = np.hstack(delta_u)

        inv_V = np.linalg.inv(V_1 + V_2);

        V_aux = np.matmul(np.matmul(OMEGA_T, inv_V), OMEGA / 4 + np.matmul(np.matmul(V_2, OMEGA), V_1))

        identity = np.identity(2 * self.N_modes);

        # V_temp = np.linalg.pinv(np.matmul(V_aux,OMEGA))                         # Trying to bypass singular matrix inversion ! I probably shouldnt do this...
        # F_tot_4 = np.linalg.det( 2*np.matmul(sqrtm(identity + matrix_power(V_temp                ,+2)/4) + identity, V_aux) );
        F_tot_4 = np.linalg.det(
            2 * np.matmul(sqrtm(identity + matrix_power(np.matmul(V_aux, OMEGA), -2) / 4) + identity, V_aux));

        F_0 = (F_tot_4.real / np.linalg.det(V_1 + V_2)) ** (
                1.0 / 4.0);  # We take only the real part of F_tot_4 as there can be a residual complex part from numerical calculations!

        F = F_0 * np.exp(-np.matmul(np.matmul(delta_u_T, inv_V), delta_u) / 4);  # Fidelity
        return F

        # Gaussian unitaries (applicable to two mode states)

    def beam_splitter(self, tau, modes=[0, 1]):
        """
        Apply a beam splitter transformation to pair of modes in a multimode gaussian state
        
        ARGUMENT:
           tau   - transmissivity of the beam splitter
           modes - indexes for the pair of modes which will receive the beam splitter operator 
        """

        # if not (isinstance(tau, list) or isinstance(tau, np.ndarray)):          # Make sure the input variables are of the correct type
        #     tau = [tau]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes,
                                                                                       range)):  # Make sure the input variables are of the correct type
            modes = [modes]

        assert len(modes) == 2, "Unable to decide which modes to apply beam splitter operator nor by how much"

        BS = np.eye(2 * self.N_modes)
        i = modes[0]
        j = modes[1]

        # B = np.sqrt(tau)*np.identity(2)
        # S = np.sqrt(1-tau)*np.identity(2)

        # BS[2*i:2*i+2, 2*i:2*i+2] = B
        # BS[2*j:2*j+2, 2*j:2*j+2] = B

        # BS[2*i:2*i+2, 2*j:2*j+2] =  S
        # BS[2*j:2*j+2, 2*i:2*i+2] = -S

        ##########################################
        cos_theta = np.sqrt(tau)
        sin_theta = np.sqrt(1 - tau)

        BS[2 * i, 2 * i] = cos_theta
        BS[2 * i + 1, 2 * i + 1] = cos_theta
        BS[2 * j, 2 * j] = cos_theta
        BS[2 * j + 1, 2 * j + 1] = cos_theta

        BS[2 * i, 2 * j] = +sin_theta
        BS[2 * j, 2 * i] = -sin_theta

        BS[2 * i + 1, 2 * j + 1] = +sin_theta
        BS[2 * j + 1, 2 * i + 1] = -sin_theta
        ##########################################

        BS_T = np.transpose(BS)

        self.R = np.matmul(BS, self.R)
        self.V = np.matmul(np.matmul(BS, self.V), BS_T)

        return BS

    def beam_splitter_neg(self, tau, modes=[0, 1]):
        """
        Apply a beam splitter transformation to pair of modes in a multimode gaussian state
        
        ARGUMENT:
           tau   - transmissivity of the beam splitter
           modes - indexes for the pair of modes which will receive the beam splitter operator 
        """

        # if not (isinstance(tau, list) or isinstance(tau, np.ndarray)):          # Make sure the input variables are of the correct type
        #     tau = [tau]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes,
                                                                                       range)):  # Make sure the input variables are of the correct type
            modes = [modes]

        assert len(modes) == 2, "Unable to decide which modes to apply beam splitter operator nor by how much"

        BS = np.eye(2 * self.N_modes)
        i = modes[0]
        j = modes[1]

        # B = np.sqrt(tau)*np.identity(2)
        # S = np.sqrt(1-tau)*np.identity(2)

        # BS[2*i:2*i+2, 2*i:2*i+2] = B
        # BS[2*j:2*j+2, 2*j:2*j+2] = B

        # BS[2*i:2*i+2, 2*j:2*j+2] =  S
        # BS[2*j:2*j+2, 2*i:2*i+2] = -S

        ##########################################
        cos_theta = np.sqrt(tau)
        sin_theta = -np.sqrt(1 - tau)

        BS[2 * i, 2 * i] = cos_theta
        BS[2 * i + 1, 2 * i + 1] = cos_theta
        BS[2 * j, 2 * j] = cos_theta
        BS[2 * j + 1, 2 * j + 1] = cos_theta

        BS[2 * i, 2 * j] = +sin_theta
        BS[2 * j, 2 * i] = -sin_theta

        BS[2 * i + 1, 2 * j + 1] = +sin_theta
        BS[2 * j + 1, 2 * i + 1] = -sin_theta
        ##########################################

        BS_T = np.transpose(BS)

        self.R = np.matmul(BS, self.R)
        self.V = np.matmul(np.matmul(BS, self.V), BS_T)

    def two_mode_squeezing(self, r, phi, modes=[0, 1]):
        """
        Apply a two mode squeezing operator  in a gaussian state
        r - squeezing intensity
        phi - squeezing phase
        
        ARGUMENT:
           r - ampllitude for the two-mode squeezing operator
           phi - squeezing phase (usually does not contribute to entanglement)
        """

        # if not (isinstance(r, list) or isinstance(r, np.ndarray)):              # Make sure the input variables are of the correct type
        #     r = [r]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes,
                                                                                       range)):  # Make sure the input variables are of the correct type
            modes = [modes]

        assert len(modes) == 2, "Unable to decide which modes to apply two-mode squeezing operator nor by how much"

        S2 = np.eye(2 * self.N_modes)
        i = modes[0]
        j = modes[1]

        S0 = np.cosh(r) * np.identity(2)
        S1 = np.array([[np.sinh(r) * np.cos(phi), np.sinh(r) * np.sin(phi)],
                       [np.sinh(r) * np.sin(phi), -np.sinh(r) * np.cos(phi)]])

        S2[2 * i:2 * i + 2, 2 * i:2 * i + 2] = S0
        S2[2 * j:2 * j + 2, 2 * j:2 * j + 2] = S0

        S2[2 * i:2 * i + 2, 2 * j:2 * j + 2] = S1
        S2[2 * j:2 * j + 2, 2 * i:2 * i + 2] = S1

        # S2 = np.block([[S0, S1], [S1, S0]])
        S2_T = np.transpose(S2)

        self.R = np.matmul(S2, self.R)
        self.V = np.matmul(np.matmul(S2, self.V), S2_T)

        return S2

    # Generic multimode gaussian unitary
    def apply_unitary(self, S, d):
        """
        Apply a generic gaussian unitary on the gaussian state
        
        ARGUMENTS:
            S,d - affine symplectic map (S, d) acting on the phase space, equivalent to gaussian unitary
        """
        assert np.all(np.isreal(d)), "Error when applying generic unitary, displacement d is not real!"

        S_is_symplectic = np.allclose(np.matmul(np.matmul(np.abs(S), self.Omega), np.transpose(np.abs(S))), self.Omega,
                                      1e-7)

        # assert S_is_symplectic , "Error when applying generic unitary, unitary S is not symplectic!"

        self.R = np.matmul(S, self.R) + d
        self.V = np.matmul(np.matmul(S, self.V), np.transpose(S))

    # Generic multimode gaussian unitary in terms of Bogoliubov coefficients
    def apply_Bogoliubov_unitary(self, S):
        """
        Apply a generic gaussian unitary on the gaussian state in terms of the Bogoliubov coefficients
        by changing first the basis and then performing the unitary transformation

        ARGUMENTS:
            S - Bogoliubov transformation matrix
        """

        B_one = np.array([[(1 + 0j) / np.sqrt(2), (0 + 1j) / np.sqrt(2)],
                          [(1 + 0j) / np.sqrt(2), (0 - 1j) / np.sqrt(2)]],
                         dtype=complex)  # To transform the matrix S into quadrature basis we create B
        B = np.kron(np.eye(self.N_modes, dtype=int), B_one)
        Sr = np.dot(np.linalg.inv(B), np.dot(S, B))
        Sr = np.real(Sr)
        self.apply_unitary(Sr, [0.])

        # Method to introduce attenuation

    def attenuation(self, eta):
        """
        Apply to the covariance matrix an attenuation of value 1-eta, or efficiency eta

        ARGUMENTS:
            eta - efficiency
        """

        assert np.isreal(eta) and eta >= 0 and eta <= 1, "Efficiency must be a real number between 0 and 1"

        self.V = eta * self.V + (1 - eta) * np.eye(2 * self.N_modes)


##############################################################################################################################
##############################################################################################################################

# Auxiliary methods


def tensor_product(rho_list):
    """ Given a list of gaussian states,
    # calculates the tensor product of all of them and store it in a new state
    #
    # PARAMETERS:
    #    rho_array - array of gaussian_state (multimodes) ex: [state1, state2, ...]
    #
     CALCULATES:
        rho - multimode gaussian_state with all of the input states
    """

    R_final = rho_list[0].R;  # First moments of resulting state is the same of rho_A
    V_final = rho_list[0].V;  # First block diagonal entry is the CM of rho_A

    for rho in rho_list[1:]:  # Loop through each state that is to appended
        R_final = np.vstack((R_final, rho.R))  # Create its first moments
        V_final = block_diag(V_final, rho.V)

    return Gaussian_state(R_final, V_final)


def elementary_states(type, modes_parameters):
    """"
    Create an elementary state of type thermal, coherent or squeezed with different parameters for each mode

    Inputs:
        - type: string with the type of state: thermal, coherent or squeezed
        - modes_parameters: array with len=number of modes and parameters for each mode

    Output:
        -Gaussian state of type "type" and number of modes len(modes_parameters)
    """

    assert isinstance(type, str), "type of gaussian state should be a string"
    assert len(modes_parameters) > 0, "at least one parameter should be said"

    single_mode_gs = []
    if (str(type) == "thermal"):
        for par in modes_parameters:
            single_mode_gs.append(Gaussian_state("thermal", 1, par))
    elif (str(type) == "coherent"):
        for par in modes_parameters:
            single_mode_gs.append(Gaussian_state("coherent", 1, par))
    elif (str(type) == "squeezed"):
        for par in modes_parameters:
            single_mode_gs.append(Gaussian_state("squeezed", 1, par))
    else:
        raise ValueError("Unrecognized gaussian state name, please type should be thermal, coherent or squeezed")

    return tensor_product(single_mode_gs)


def BasisChange(S, type):
    """
    Inputs: -A scattering matrix
            -The original type of the matrix (1 for S in terms of bogoliubov coefs or 0 for S in terms of quadratures)
    Outputs: -The scattering matrix in the other basis
    """

    assert isinstance(S, np.ndarray) and S.shape[0] == S.shape[1], "S matrix is not an array or is not an square matrix"
    assert isinstance(type, int) and (type == 0 or type == 1), "Second argument must be 0 or 1"

    modes = int(S.shape[0] / 2)
    B_one = np.array([[(1 + 0j) / np.sqrt(2), (0 + 1j) / np.sqrt(2)],
                      [(1 + 0j) / np.sqrt(2), (0 - 1j) / np.sqrt(2)]],
                     dtype=complex)  # To transform the matrix S into quadrature basis we create B
    B = np.kron(np.eye(modes, dtype=int), B_one)

    if type == 0:
        return np.dot(B, np.dot(S, np.linalg.inv(B)))

    elif type == 1:
        return np.real(np.dot(np.linalg.inv(B), np.dot(S, B)))


def LN_generalized(S, InState, subsA, subsB, *args):
    """
    Inputs: -Scattering matrix in terms of the Bogoluibov coefficients (in annhilation/creation basis (S) if 1 is introduced as args)
        -A Gaussian State to apply transformation on it (InState)  
        -An array telling which are the modes from the subsystem A (subsA)
        -An array telling which are the modes from the subsystem B (subsB) 
 
        The number of modes N should be the sum of len(subsA)+len(subsB) and the dimensions of the scattering matrix 2N, as well as 
        the gaussian state should be of N modes (eliminated for now to allow LN between subsystems)
    """

    assert isinstance(InState, Gaussian_state), "In State is not an object of class Gaussian State"
    assert isinstance(S, np.ndarray) and S.shape[0] == S.shape[1], "S matrix is not an array or is not an square matrix"
    assert InState.N_modes == (S.shape[0] / 2), "Scaterring matrix of different order as 2*(Number of modes) of InState"
    # assert len(subsA)+len(subsB)==InState.N_modes, "Subsystems A and B do not sum the same number of modes as in InState"

    Sr = S
    N = InState.N_modes  # To know how many modes we have
    if len(args) > 0 and args[0] == 1:
        Sr = BasisChange(S, 1)  # We change the scattering matrix to the quadrature basis
    InState.apply_unitary(Sr, [0.])  # We apply the scattering matrix (with displacement 0)
    if len(args) > 1:
        eta = args[1]
        InState.attenuation(eta)
    return (InState.logarithmic_negativity(subsA, subsB))


def Is_Sympletic(S, type, atol=1e-5):
    """
    Inputs: -A scattering matrix
            -The type of the matrix (1 for S in terms of bogoliubov coefs or 0 for S in terms of quadratures)
    """

    assert isinstance(S, np.ndarray) and S.shape[0] == S.shape[1], "S matrix is not an array or is not an square matrix"
    assert isinstance(type, int) and (type == 0 or type == 1), "Second argument must be 0 or 1"

    modes = int(S.shape[0] / 2)

    if type == 0:
        omega1 = np.array([[0, 1], [-1, 0]])  # Auxiliar variable for 1 mode omega
        Omega = np.kron(np.eye(modes, dtype=int), omega1)
        return np.allclose(np.matmul(np.matmul(np.abs(S), Omega), np.transpose(np.abs(S))), Omega, 1e-5)

    elif type == 1:
        for I in range(0, modes):
            for J in range(0, modes):  # Go through I and J and sum over K the two Bogoliubov relations
                rel1 = 0
                rel2 = 0
                rel3 = 0
                rel4 = 0
                for K in range(0, modes):
                    rel1 = rel1 + S[2 * K][2 * I] * S[2 * K + 1][2 * J + 1] - S[2 * K + 1][2 * I] * S[2 * K][2 * J + 1]
                    rel2 = rel2 + S[2 * K][2 * I] * S[2 * K + 1][2 * J] - S[2 * K + 1][2 * I] * S[2 * K][2 * J]
                    rel3 = rel3 + S[2 * I + 1][2 * K + 1] * S[2 * J][2 * K] - S[2 * I + 1][2 * K] * S[2 * J][2 * K + 1]
                    rel4 = rel4 + S[2 * I + 1][2 * K + 1] * S[2 * J + 1][2 * K] - S[2 * I + 1][2 * K] * S[2 * J + 1][
                        2 * K + 1]

                if I == J:
                    if not np.isclose(np.abs(rel1), 1.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} first Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel1)}")
                        return False
                    if not np.isclose(np.abs(rel2), 0.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} second Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel2)}")
                        return False
                    if not np.isclose(np.abs(rel3), 1.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} third Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel3)}")
                        return False
                    if not np.isclose(np.abs(rel4), 0.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} fourth Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel4)}")
                        return False
                else:
                    if not np.isclose(np.abs(rel1), 0.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} first Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel1)}")
                        return False
                    if not np.isclose(np.abs(rel2), 0.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} second Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel2)}")
                        return False
                    if not np.isclose(np.abs(rel3), 0.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} third Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel3)}")
                        return False
                    if not np.isclose(np.abs(rel4), 0.0, atol=atol):
                        print(f"for I = {I + 1}, J = {J + 1} fourth Bogoliubov relation does not hold ")
                        print(f"Relation happens to be {np.abs(rel4)}")
                        return False
        return True

    else:
        print(f"type of scattering matrix not specified")
