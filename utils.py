import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from CF import config_parameters as c_p

class CustomFunctions:
    """ Collection of useful methods for this project.
    
    """
    def num_from_interval_to_interval(self, number, min_int_1, max_int_1,
                                      min_int_2, max_int_2):
        """ Rescales a number from one interval to another.

        Parameters
        ----------
        number : float
        a number to rescale
        min_int_1 : float
        minimum of interval to rescale from
        max_int_1 : float
        minimum of interval to rescale from
        min_int_2 : float
        minimum of interval to rescale to
        max_int_2 : float
        minimum of interval to rescale to
        """
        interval_1_len = max_int_1 - min_int_1
        interval_2_len = max_int_2 - min_int_2
        return interval_2_len / interval_1_len * (number - min_int_1) + min_int_2
    
    def normalize_band(self, bands_dataset, E_min=0.0, E_max=0.0):
        for bands in bands_dataset:
            for band_idx in range(bands.shape[1]):
                for E in bands[:,band_idx]:
                    if E_max < E:
                        E_max = E
                    if E_min > E:
                        E_min = E
        
        for bands in bands_dataset:
            for band_idx in range(bands.shape[1]):
                for E_idx in range(bands[:,band_idx].shape[0]):
                    bands[E_idx][band_idx] = self.num_from_interval_to_interval(bands[E_idx][band_idx], E_min, E_max, 0, 1)
    
        return E_min, E_max
    
    def get_means(self, bands_dataset):
        means = [0.0, 0.0, 0.0]
        for bands in bands_dataset:
            for band_idx in range(bands.shape[1]):
                means[band_idx] += bands[:,band_idx].sum() / bands.shape[0]
        for band_idx in range(bands.shape[1]):
            means[band_idx] /= bands_dataset.shape[0]
        return means
        
    def get_standard_deviations(self, bands_dataset, means):
        standard_deviations = [0.0, 0.0, 0.0]
        for bands in bands_dataset:
            for band_idx in range(bands.shape[1]):
                for E in bands[:,band_idx]:
                    standard_deviations[band_idx] += (E-means[band_idx])**2
        for band_idx in range(bands.shape[1]):
            standard_deviations[band_idx] /= (bands_dataset.shape[0] * bands_dataset.shape[1])
        return np.sqrt(standard_deviations)
    
    def plot_band(self, material, bands):
        grid_k, critical_points = HexagonalLattice.k_GG_path_hexagonal(material.a0)
        plotter = Plotting(grid_k, critical_points)
        plotter.plot_Ek(bands)
    
    def compare_output_vs_target(self, material_output, material_target, bands_output, bands_target):
        grid_k, critical_points = HexagonalLattice.k_GG_path_hexagonal(material_output.a0)
        plotter = Plotting(grid_k, critical_points)
        
        bands_o = GeneratingData.calculate_band(material_output, grid_k)
        bands_t = GeneratingData.calculate_band(material_target, grid_k)
        
        plotter.plot_Ek_output_target(bands_o, bands_target)
        
        
cf = CustomFunctions()


class AtomicUnits:
    """Class storing atomic units.

    All variables, arrays in simulations are in atomic units.

    Attributes
    ----------
    Eh : float
        Hartree energy (in meV)
    Ah : float
        Bohr radius (in nanometers)
    Th : float
        time (in picoseconds)
    Bh : float
        magnetic induction (in Teslas)
    """
    # atomic units
    Eh=27211.4 # meV
    Ah=0.05292 # nm
    Th=2.41888e-5 # ps
    Bh=235051.76 # Teslas

au = AtomicUnits()


class TMDCmaterial:
    """ Class containing lattice model parameters.

    Attributes
    ----------
    a0 : float
        material lattice constant (in nanometers)
    dim : int
        Hilbert space (sub)dimension: no of orbitals x spin degree = 3 x 2,
        dimension of the whole state-space = dim*N, where N is a no of lattice nodes
    dim2 : int
        squared dim: dim2 = dim*dim
    V.. : float
        bond integrals
    e. : float
        tight-binding onsite energies
    """
    def __init__(self, a0, Vd2sigma, Vd2pi, Vd2delta, e0, e1, e2):
        self.a0 = a0/au.Ah
        self.dim = 3
        self.dim2 = self.dim*self.dim
        # hoppings
        self.Vd2sigma = Vd2sigma/au.Eh
        self.Vd2pi = Vd2pi/au.Eh
        self.Vd2delta = Vd2delta/au.Eh
        # onsite energy
        self.e0 = e0/au.Eh
        self.e1 = e1/au.Eh
        self.e2 = e2/au.Eh
        self.diag = np.array([self.e0,self.e1,self.e2])


class HexagonalLattice:
    """ Class storing lattice structure definition.

    Attributes
    ----------
    lattice_vectors : List[ndarray]
        lattice vectors definition
    """
    def __init__(self, a0):
        self.a0 = a0
        R1 = np.array([1., 0.])
        R2 = np.array([1., np.sqrt(3.)])/2.
        R3 = np.array([-1., np.sqrt(3.)])/2.
        R1 *= self.a0
        R2 *= self.a0
        R3 *= self.a0
        R4 = -R1
        R5 = -R2
        R6 = -R3
        self.lattice_vectors = [R1, R2, R3, R4, R5, R6]
        
    def k_GG_path_hexagonal(a0):
        K = np.pi*4./3./a0
        M = K*3./2
        G = K*np.sqrt(3.)/2.
        critical_points = [(r'$\Gamma$', 0.), ('K', K), ('M', M), (r'$\Gamma$', M+G)]
        k_GK = [[x, 0.] for x in np.arange(0, K, 0.01)] # k varying from Gamma to K point within the BZ
        k_KM = [[x, 0.] for x in np.arange(K, M, 0.01)] # k varying from K to M point within the BZ
        k_MG = [[M, y]  for y in np.arange(0, G, 0.01)] # k varying from M to Gamma point within the BZ
        k_GG = np.concatenate((k_GK, k_KM, k_MG)) # full path within the BZ
        return k_GG, critical_points
    
HexagonalLattice.k_GG_path_hexagonal = staticmethod(HexagonalLattice.k_GG_path_hexagonal)


class FlakeModel:
    """ Collection of methods for creating the flake model.

    Attributes
    ----------
    material : TMDCmaterial
        flake material parameters
    lattice : Lattice
        object containing lattice structure definition
    """
    def __init__(self, material, lattice):
        self.material = material
        self.lattice_vectors = lattice.lattice_vectors
        self.hopping_matrices = self.hopping_matrices()

    def hopping_matrices(self):
        """ Creates hopping matrices in all nns directions.

        Returns
        -------
        hopping_matrices : List[ndarray]
            List of 3 x 3 Hermitian hopping matrices
        """
        hopping_matrices = []
        for Ri in self.lattice_vectors:
            hopping_matrices.append(self.fill_in_hopping_matrix(Ri[0]/self.material.a0, Ri[1]/self.material.a0, 0.0))
        
        return hopping_matrices

    def tb_hamiltonian(self, k_vector, **kwargs):
        """ Creates k-dependent Hamiltonian for the infinite lattice.

        Parameters
        ----------
        k_vector : ndarray
            2d wave vector
        lattice_vectors (optional) : list[ndarray]
            list of 2d lattice vectors,
            if None self.lattice_vectors will be used

        Returns
        -------
        tb_hamiltonian : ndarray
            3 x 3 tight-binding Hamiltonian
        """
        lattice_vectors = kwargs.get('lattice_vectors', None)
        if lattice_vectors:
            self.lattice_vectors = lattice_vectors
        tb_hamiltonian = np.zeros((self.material.dim, self.material.dim), dtype=np.complex128)
        np.fill_diagonal(tb_hamiltonian, self.material.diag)
        for i_direction, lattice_vector in enumerate(self.lattice_vectors):
            phase = np.dot(k_vector, lattice_vector)
            tb_hamiltonian += self.hopping_matrices[i_direction]*(1j*np.sin(phase) + np.cos(phase))
        return tb_hamiltonian

    def solve_eigenproblem(self, hamiltonian, eigenvalues_only=False):
        """ Solves eigenproblem for the Hamiltonian matrix.

        Parameters
        ----------
        hamiltonian : ndarray
            2d hermitian input matrix
        eigenvalues_only : bool
            if True, eigenvectors will not be calculated and returned

        Returns
        -------
        eigenvalues : ndarray
            1d array of resulting eigenvalues
        eigenvectors : ndarray
            2d array containing dim x 1d eigenvectors
        """
        return eigh(hamiltonian, eigvals_only=eigenvalues_only)
    
    def SKz21z2(self, l, m, n):
        mt = self.material
        return 3.0*n**2*(l**2+m**2)*mt.Vd2pi+0.75*(l**2+m**2)**2*mt.Vd2delta+(0.5*(-l**2-m**2)+n**2)**2*mt.Vd2sigma
    
    def SKxy1z2(self, l, m, n):
        mt = self.material
        return np.sqrt(3.0)*(-2.0*l*m*n**2*mt.Vd2pi+0.5*l*m*(1+n**2)*mt.Vd2delta+l*m*(0.5*(-l**2-m**2)+n**2)*mt.Vd2sigma)
    
    def SKx2y21z2(self, l, m, n):
        mt = self.material
        return np.sqrt(3.0)*(n**2*(-l**2+m**2)*mt.Vd2pi+0.25*(l**2-m**2)*(1+n**2)*mt.Vd2delta+0.5*(l**2-m**2)*(0.5*(-l**2-m**2)+n**2)*mt.Vd2sigma)
    
    def SKxy1xy(self, l, m, n):
        mt = self.material
        x=(l**2+m**2-4.0*l**2*m**2)*mt.Vd2pi+(l**2*m**2+n**2)*mt.Vd2delta+3.0*l**2*m**2*mt.Vd2sigma
        return x
    
    def SKxy1x2y2(self, l, m, n):
        mt = self.material
        return 2*l*m*(-l**2+m**2)*mt.Vd2pi+0.5*l*m*(l**2-m**2)*mt.Vd2delta+1.5*l*m*(l**2-m**2)*mt.Vd2sigma
    
    def SKx2y21x2y2(self, l, m, n):
        mt = self.material
        return (l**2+m**2-(l**2-m**2)**2)*mt.Vd2pi+(0.25*(l**2-m**2)**2+n**2)*mt.Vd2delta+0.75*(l**2-m**2)**2*mt.Vd2sigma
    
    def fill_in_hopping_matrix(self, l, m, n):
        mt = self.material
        h_matrix = np.zeros((mt.dim, mt.dim), dtype=np.float64)
        h_matrix[0, 0] = self.SKz21z2(l, m, n); h_matrix[0, 1] = self.SKxy1z2(-l, -m, -n); h_matrix[0, 2] = self.SKx2y21z2(-l, -m, -n)
        h_matrix[1, 0] = self.SKxy1z2(l, m, n); h_matrix[1, 1] = self.SKxy1xy(l, m, n); h_matrix[1, 2] = self.SKxy1x2y2(l, m, n)
        h_matrix[2, 0] = self.SKx2y21z2(l, m, n); h_matrix[2, 1] = self.SKxy1x2y2(-l, -m, -n); h_matrix[2, 2] = self.SKx2y21x2y2(l, m, n)
        return h_matrix
    
class Plotting:
    """ Plotting utils.
    
    Attributes
    ----------
    grid_k : List[List]
        2d list containing full path within the BZ
    critical_points : List[Tuple]
        list of tuples containing critical points`s names and their coordinates
    """
    def __init__(self, grid_k, critical_points, directory=None):
        self.grid_k = grid_k
        self.critical_points = critical_points
        if directory:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'

    def plot_Ek(self, Ek, x_label='k (nm$^{-1}$)', y_label='E (meV)'):
        """ Plots dispersion relation.

        Parameters
        ----------
        Ek : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        _, ax = plt.subplots()
        ax.axes.set_aspect(.003)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        
        # plot dispersion relation
        Ek = np.array(Ek)
        for band_idx in range(Ek.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek[:,band_idx]*au.Eh, label='Band' + str(band_idx))

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01            
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 100))
             ax.axvline(x=position_k, linestyle='--', color='black')
        
    def plot_Ek_output_target(self, Ek_output, Ek_target, x_label='k (nm$^{-1}$)', y_label='E (meV)'):
        """ Plots dispersion relations for
        two given lists of bands.

        Parameters
        ----------
        Ek_. : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(0.003)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        Ek_output = np.array(Ek_output)
        Ek_target = np.array(Ek_target)
        for band_idx in range(Ek_output.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output[:,band_idx]*au.Eh, color='red', label='Output band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx]*au.Eh, color='blue', label='Target band')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01            
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 100))
             ax.axvline(x=position_k, linestyle='--', color='black')


class ManagingFiles:
    """ Utils to save and load data in .npy format.
    
    """
    def __init__(self, directory_save=None, directory_load=None):
        if directory_save:
            self.directory_save = os.path.join('./', directory_save)
            os.makedirs(directory_save, exist_ok=True)
        else:
            self.directory_save = './'
        if directory_load:
            self.directory_load = os.path.join('./', directory_load)
            os.makedirs(directory_load, exist_ok=True)
        else:
            self.directory_load = './'
            
    def save_bands_dataset(self, materials_dataset, bands_dataset, filename):
        """ Appends parameters and band to a file.

        Parameters
        ----------
        material : TMDCmaterial
            flake material parameters
        bands : List[List[array]]
            2d list of lists of arrays of eigenvalues
        filename : string
            name of a file to save the data to
        """
        filename = os.path.join(self.directory_save, filename)
        with open(filename, 'ab') as f:
            np.save(f, materials_dataset)
            np.save(f, bands_dataset)
        
    def load_bands_dataset(self, filename):
        """ Creates a list of flake materials parameters
            and a list of bands by loading data
            from a file.

        Parameters
        ----------
        size_of_data : int
            number of independent TDMC parameters,
            corresponding to independent bands,
            saved to a file.
        filename : string
            name of a file to load the data from
        """
        filename = os.path.join(self.directory_load, filename)
        with open(filename, 'rb') as f:
            materials_dataset = np.load(f, allow_pickle = True)
            bands_dataset = np.load(f, allow_pickle=True)
        return materials_dataset, bands_dataset
    
    def load_and_interpolate_bands(self, filename, interpolate_from, interpolate_to):
        filename = os.path.join(self.directory_load, filename)
        df = pd.read_csv(filename)
        bands = np.array([(np.array(x) * 1000) / au.Eh for x in  df.values.tolist()])
        bands = np.swapaxes(bands, 0, 1)
        original_interval = np.linspace(0, 1, num=interpolate_from, endpoint=True)
        interpolating_interval = np.linspace(0, 1, num=interpolate_to, endpoint=True)
        bands_fs = [ interp1d(original_interval, band) for band in bands ]
        bands_interpolated = np.array([ f(interpolating_interval) for f in bands_fs ])
        bands_interpolated = np.swapaxes(bands_interpolated, 0, 1)
        material_bands = np.array([bands_interpolated])
        material_props = [TMDCmaterial(0.319, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)]
        return material_props, material_bands
    
    def erase_data(self, filename):
        """ Erases data from a file.

        Parameters
        ----------
        filename : string
            name of a file to erase data from
        """
        filename = os.path.join(self.directory_save, filename)
        open(filename, 'w').close()


class GeneratingData:
    """ Collection of methods for generating lattice model parameters.

    Attributes
    ----------
    a0_bounds : Tuple[float]
        material lattice constant
        lower and upper bounds (in nanometers)
    V._bounds. : Tuple[float]
        bond integrals
        lower and upper bounds
    e_bounds. : Tuple[float]
        tight-binding onsite energies
        lower and upper bounds
    lso_bounds : Tuple[float]
        intinsic spin-orbit energy (in meV)
        lower and upper bounds
    file_manager : ManagingFiles
        object responsible for saving to,
        loading to or erasing data from a file.
    """
    def __init__(self, a0_bounds, Vd2sigma_bounds, Vd2pi_bounds,
                 Vd2delta_bounds, e0_bounds, e1_bounds, e2_bounds, file_manager):
        self.a0_bounds = a0_bounds
        self.Vd2sigma_bounds = Vd2sigma_bounds
        self.Vd2pi_bounds = Vd2pi_bounds
        self.Vd2delta_bounds = Vd2delta_bounds
        self.e0_bounds = e0_bounds
        self.e1_bounds = e1_bounds
        self.e2_bounds = e2_bounds
        self.file_manager = file_manager
        
    def generate_data(self, size_of_data):
        """ Randomizes flake material parameters
            and calculates bands, until
            a dataset of size size_of_data is obtained.

        Parameters
        ----------
        filename : string
            name of a file to load the data from
        size_of_data : int
            size of a dataset to generate.
        """
        materials_dataset = []
        bands_dataset = []
        i = 0
        while i < size_of_data:
            material_parameters = np.random.rand(7)
            a0 = cf.num_from_interval_to_interval(material_parameters[0], 0, 1, self.a0_bounds[0], self.a0_bounds[1])
            Vd2sigma = cf.num_from_interval_to_interval(material_parameters[1], 0, 1, self.Vd2sigma_bounds[0], self.Vd2sigma_bounds[1])
            Vd2pi = cf.num_from_interval_to_interval(material_parameters[2], 0, 1, self.Vd2pi_bounds[0], self.Vd2pi_bounds[1])
            Vd2delta = cf.num_from_interval_to_interval(material_parameters[3], 0, 1, self.Vd2delta_bounds[0], self.Vd2delta_bounds[1])
            e0 = cf.num_from_interval_to_interval(material_parameters[4], 0, 1, self.e0_bounds[0], self.e0_bounds[1])
            e1 = cf.num_from_interval_to_interval(material_parameters[5], 0, 1, self.e1_bounds[0], self.e1_bounds[1])
            e2 = cf.num_from_interval_to_interval(material_parameters[6], 0, 1, self.e2_bounds[0], self.e2_bounds[1])
            material = TMDCmaterial(a0, Vd2sigma, Vd2pi, Vd2delta, e0, e1, e2)
            k_GG, _ = HexagonalLattice.k_GG_path_hexagonal(material.a0)
            bands = GeneratingData.calculate_band(material, k_GG)
            
            frac = i / size_of_data
            
            if (frac < c_p.frac_of_band_gaps):
                if (self.is_band_gap_acc(bands)):
                    materials_dataset.append(material)
                    bands_dataset.append(bands)
                    i = i + 1
            else:
                materials_dataset.append(material)
                bands_dataset.append(bands)
                i = i + 1
                    
        return np.asarray(materials_dataset), np.asarray(bands_dataset)
    
    def calculate_band(material, k_GG):
        """ Given flake material parameters
        calculates the dispersion relation
        along Gamma - K - M - Gamma path.

        Parameters
        ----------
        material : TMDCmaterial
            flake material parameters
        k_GG : List[List]
            2d list containing full path within the BZ
        """
        lattice = HexagonalLattice(material.a0)
        flake = FlakeModel(material, lattice)

        bands = []
        for k in k_GG:
            hamiltonian = flake.tb_hamiltonian(k)
            eigenvalues = flake.solve_eigenproblem(hamiltonian, eigenvalues_only=True)
            bands.append(eigenvalues)
        return bands
    
    def is_band_gap_min_max(self, bands):
        bs = np.swapaxes(bands, 0, 1)
        if ((np.amin(bs[1]) - np.amax(bs[0])) > 0.005
            and (np.amin(bs[2]) - np.amax(bs[0])) > 0.005):
            return True
        else:
            return False
        
    def is_band_gap_acc(self, bands):
        bands = np.sort(bands, axis=None)
        bands = np.diff(bands)
        no_energies = bands.shape[0]
        for idx, diff in enumerate(bands):
            if diff > c_p.band_gap and idx > 0.33 * no_energies and idx < 0.66 * no_energies:
                return True
        return False
        
GeneratingData.calculate_band = staticmethod(GeneratingData.calculate_band)