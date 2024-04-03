import os
import numpy as np
import pyvista
from pyvista import CellType
from crackpy.structure_elements import data_files
from crackpy.structure_elements.material import Material


class InputData:
    """Class for the import and transformation of input data from DIC nodemaps.

    Methods:
        * read_header - method to import metadata from header (if ``data_file`` is provided)
        * read_data_file - method to import input data from file (if ``data_file`` is provided)
        * set_connection_file - method to set connection file path

        * set_data_manually - set physical data manually instead of from nodemap file
        * set_data_file - set data file path
        * transform_data - with a coordinate shift and angle rotation

        * calc_eps_vm - calculate Von Mises strain (stored as attribute)
        * calc_stresses - calculate stresses using material law
        * calc_facet_size - calculate the face size of DIC data
        * to_vtk - convert nodemap data to a PyVista object and save it in vtk format

    """

    def __init__(self, nodemap: data_files.Nodemap or None = None, meta_keywords: dict or None = None):
        """Initialize input data class by reading metadata and data and calculating eps_vm (if data_file is provided).

        Args:
            nodemap: None or obj of class Nodemap (if provided the methods ``read_data_file``,
                                            ``read_header``, and ``calc_eps_vm``` are run upon initialization)
            meta_keywords: keywords to read header
        """

        if nodemap is not None:
            self.nodemap_folder = nodemap.folder
            self.nodemap_name = nodemap.name
            self.data_file = os.path.join(self.nodemap_folder, self.nodemap_name)
            self.nodemap_structure = nodemap.structure
        else:
            self.nodemap_folder = None
            self.nodemap_name = None
            self.data_file = None
            self.nodemap_structure = data_files.NodemapStructure()

        self.meta_keywords = meta_keywords
        self.connection_file = None

        # input data attributes
        self.facet_id = None
        self.coor_x = None
        self.coor_y = None
        self.coor_z = None
        self.disp_x = None
        self.disp_y = None
        self.disp_z = None
        self.eps_x = None
        self.eps_y = None
        self.eps_xy = None
        self.eps_vm = None
        self.sig_x = None
        self.sig_y = None
        self.sig_xy = None
        self.sig_vm = None
        self.eps_1 = None
        self.eps_2 = None
        self.sig_1 = None
        self.sig_2 = None
        self.connections = None

        # meta data attributes
        self.meta_attributes = [
            'force',
            'cycles',
            'displacement',
            'potential',
            'cracklength',
            'time',
            'dms_1', 'dms_2',
            'x', 'y', 'z',
            'alignment_translation_x',
            'alignment_translation_y',
            'alignment_translation_z'
        ]
        for attribute in self.meta_attributes:
            setattr(self, attribute, None)

        # methods called when initialized
        if self.data_file is not None:
            self.read_header(meta_attributes_to_keywords=self.meta_keywords)
            self.read_data_file()

    def set_data_file(self, data_file: str):
        """Set data file path.

        Args:
            data_file: name of data file
        """
        self.data_file = data_file

    def set_connection_file(self, connection_file: str, folder: str):
        """Set connection file path.

        Args:
            connection_file: name of connection file
            folder: folder of connection file

        """
        connection_file_path = os.path.join(folder, connection_file)
        np_df = np.genfromtxt(connection_file_path, dtype=int, delimiter=';', skip_header=1)
        self.connections = self._cut_none_elements(np_df)

    def read_header(self, meta_attributes_to_keywords: dict = None):
        """Get meta data by reading from header.

        Args:
            meta_attributes_to_keywords: dictionary with meta data attributes as keys and corresponding keyword in header as values
                                         if None, the class attributes are used as keywords

        """
        if meta_attributes_to_keywords is None:
            # set default meta data keywords
            meta_attributes_to_keywords = {}
            for attribute in self.meta_attributes:
                meta_attributes_to_keywords[attribute] = attribute

        with open(self.data_file, 'r', errors='ignore') as input_data:
            for line in input_data:
                if not line.startswith('#'):
                    break
                for meta_attr, meta_key in meta_attributes_to_keywords.items():
                    if '# ' + meta_key in line:
                        meta_stripped = line.split(':')[-1].strip()  # can be str(float) or 'None'
                        if meta_stripped == 'None':
                            meta_value = None
                        else:
                            meta_value = float(meta_stripped)

                        # set class instance attribute
                        setattr(self, meta_attr, meta_value)

                        meta_attributes_to_keywords.pop(meta_attr)
                        break

    def read_data_file(self):
        """Read data from nodemap file."""
        raw_inp = np.genfromtxt(self.data_file,
                                delimiter=";", encoding="windows-1252")
        np_df = np.asarray(raw_inp, dtype=np.float64)
        # cut nans (necessary since version 2020)
        nodemap_data = self._cut_nans(np_df)
        self.facet_id = nodemap_data[:, self.nodemap_structure.index_facet_id]
        self.coor_x = nodemap_data[:, self.nodemap_structure.index_coor_x]
        self.coor_y = nodemap_data[:, self.nodemap_structure.index_coor_y]
        self.coor_z = nodemap_data[:, self.nodemap_structure.index_coor_z]
        self.disp_x = nodemap_data[:, self.nodemap_structure.index_disp_x]
        self.disp_y = nodemap_data[:, self.nodemap_structure.index_disp_y]
        self.disp_z = nodemap_data[:, self.nodemap_structure.index_disp_z]

        if self.nodemap_structure.strain_is_percent:
            self.eps_x = nodemap_data[:, self.nodemap_structure.index_eps_x] / 100.0
            self.eps_y = nodemap_data[:, self.nodemap_structure.index_eps_y] / 100.0
        else:
            self.eps_x = nodemap_data[:, self.nodemap_structure.index_eps_x]
            self.eps_y = nodemap_data[:, self.nodemap_structure.index_eps_y]
        self.eps_xy = nodemap_data[:, self.nodemap_structure.index_eps_xy]
        self.calc_eps_vm()

        if self.nodemap_structure.is_fem:
            self.sig_x = nodemap_data[:, 11]
            self.sig_y = nodemap_data[:, 12]
            self.sig_xy = nodemap_data[:, 13]
            self.sig_vm = self._calc_sig_vm()

    def set_data_manually(self, coor_x: np.array, coor_y: np.array, disp_x: np.array, disp_y: np.array,
                          eps_x: np.array, eps_y: np.array, eps_xy: np.array, eps_vm: np.array = None):
        """Manually set data, e.g. for in-situ calculation in Aramis software.

        Args:
            coor_x: x coordinates
            coor_y: y coordinates
            disp_x: x displacements
            disp_y: y displacements
            eps_x: strain eps_x
            eps_y: strain eps_y
            eps_xy: strain eps_xy
            eps_vm: Von-Mises strain (if provided)

        """
        if eps_vm is None:
            df = np.asarray([coor_x, coor_y, disp_x, disp_y, eps_x, eps_y, eps_xy]).transpose()
        else:
            df = np.asarray([coor_x, coor_y, disp_x, disp_y, eps_x, eps_y, eps_xy, eps_vm]).transpose()
        df = self._cut_nans(df)

        self.coor_x = df[:, 0]  # coor_x
        self.coor_y = df[:, 1]  # coor_y
        self.disp_x = df[:, 2]  # disp_x
        self.disp_y = df[:, 3]  # disp_y
        self.eps_x = df[:, 4]  # eps_x
        self.eps_y = df[:, 5]  # eps_y
        self.eps_xy = df[:, 6]  # eps_xy
        if eps_vm is not None:
            self.eps_vm = df[:, 7]  # eps_vm
        else:
            self.eps_vm = []

    def transform_data(self, x_shift: float, y_shift: float, angle: float):
        """Transform data by shift and rotation.

        Args:
            x_shift: shift of x-coordinate
            y_shift: shift of y-coordinate
            angle: rotation angle in degrees between 0° and 360°

        """
        angle *= np.pi / 180.0  # deg to rad
        trafo_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                 [-np.sin(angle), np.cos(angle)]])
        # coordinates
        self.coor_x -= x_shift
        self.coor_y -= y_shift
        # transformation (x',y') = R * (x,y)
        self.coor_x, self.coor_y = np.dot(trafo_matrix, [self.coor_x, self.coor_y])
        self.disp_x, self.disp_y = np.dot(trafo_matrix, [self.disp_x, self.disp_y])

        # transformation eps' = R * eps * R^T for each node
        eps_x, eps_y, eps_xy = [], [], []
        for node_i, _ in enumerate(self.eps_x):
            strain_i = np.array([[self.eps_x[node_i], self.eps_xy[node_i]],
                                 [self.eps_xy[node_i], self.eps_y[node_i]]])
            strain_i = np.dot(strain_i, trafo_matrix.T)
            strain_i = np.dot(trafo_matrix, strain_i)
            eps_x.append(strain_i[0, 0])
            eps_y.append(strain_i[1, 1])
            eps_xy.append(strain_i[0, 1])
        self.eps_x = np.asarray(eps_x)
        self.eps_y = np.asarray(eps_y)
        self.eps_xy = np.asarray(eps_xy)

        # transformation sig' = R * sig * R^T for each node
        sig_x, sig_y, sig_xy = [], [], []
        for node_i, _ in enumerate(self.sig_x):
            stress_i = np.array([[self.sig_x[node_i], self.sig_xy[node_i]],
                                 [self.sig_xy[node_i], self.sig_y[node_i]]])
            stress_i = np.dot(stress_i, trafo_matrix.T)
            stress_i = np.dot(trafo_matrix, stress_i)
            sig_x.append(stress_i[0, 0])
            sig_y.append(stress_i[1, 1])
            sig_xy.append(stress_i[0, 1])
        self.sig_x = np.asarray(sig_x)
        self.sig_y = np.asarray(sig_y)
        self.sig_xy = np.asarray(sig_xy)

        # recalculate Von Mises stress and eq. strain
        self.eps_vm = self.calc_eps_vm()
        self.sig_vm = self._calc_sig_vm()

    def calc_eps_vm(self):
        """Calculate and return Von Mises equivalent strain.

        Reference: https://en.wikipedia.org/wiki/Infinitesimal_strain_theory#Equivalent_strain

        """
        # under plane stress assumption with nu = 0.5
        nu = 0.5
        eps_z = -nu / (1 - nu) * (self.eps_x + self.eps_y)

        # total strain: eps = [[eps_x, eps_xy, 0], [eps_xy, eps_y, 0], [0, 0, eps_z]]
        eps = np.array([[self.eps_x, self.eps_xy, np.zeros_like(self.eps_x)],
                        [self.eps_xy, self.eps_y, np.zeros_like(self.eps_x)],
                        [np.zeros_like(self.eps_x), np.zeros_like(self.eps_x), eps_z]]).transpose(2, 0, 1)
        # deviatoric strain: eps_dev = eps - 1/3 * tr(eps) * I
        eps_dev = eps - 1 / 3 * np.trace(eps, axis1=1, axis2=2)[:, np.newaxis, np.newaxis] * np.eye(3)

        # von Mises equivalent strain: eps_vm = sqrt(2/3 * eps_dev:eps_dev)
        self.eps_vm = np.sqrt(2 / 3 * np.sum(eps_dev ** 2, axis=(1, 2)))
        return self.eps_vm

    def calc_stresses(self, material: Material):
        """Calculates and attaches the stresses using linear elasticity with the provided parameters.

        Args:
            material: obj of class Material to get stiffness matrix

        """
        self.sig_x, self.sig_y, self.sig_xy = np.dot(material.stiffness_matrix,
                                                     [self.eps_x, self.eps_y, self.eps_xy])
        self.sig_vm = self._calc_sig_vm()

    def calc_facet_size(self) -> float:
        """Returns DIC facet size."""
        return np.min(
            np.sqrt(
                (self.coor_x[1:] - self.coor_x[0]) ** 2.0
                + (self.coor_y[1:] - self.coor_y[0]) ** 2.0
            )
        )

    def _calc_sig_vm(self):
        """Returns the Von Mises stress."""
        return np.sqrt(self.sig_x ** 2 + self.sig_y ** 2 - self.sig_x * self.sig_y + 3 * self.sig_xy ** 2)

    def _calculate_principal_tensor_components(self, tensor_components: np.array):
        """Calculate principal tensor components

        Args:
            tensor_components: array of tensor components (e.g. stress or strain) with shape (3)

        """
        xx, yy, xy = tensor_components

        tensor = np.array([[xx, xy],
                           [xy, yy]])

        # Compute the eigenvalues of the stress tensor
        eigenvalues, _ = np.linalg.eig(tensor)

        # Sort the eigenvalues to obtain the principal stresses
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        return eigenvalues_sorted

    def _calculate_principal_strains(self):
        """Calculate principal strains"""
        strains = np.stack([self.eps_x, self.eps_y, self.eps_xy], axis=1)
        principal_strains = np.apply_along_axis(self._calculate_principal_tensor_components, 1, strains)
        self.eps_1 = principal_strains[:, 0]
        self.eps_2 = principal_strains[:, 1]

    def _calculate_principal_stresses(self):
        """Calculate principal stresses"""
        stresses = np.stack([self.sig_x, self.sig_y, self.sig_xy], axis=1)
        principal_stresses = np.apply_along_axis(self._calculate_principal_tensor_components, 1, stresses)
        self.sig_1 = principal_stresses[:, 0]
        self.sig_2 = principal_stresses[:, 1]

    def _renumber_nodes(self, old_node_number: int, mapping: dict):
        """Renumber nodes according to mapping table.

        Args:
            old_node_number: node number
            mapping: mapping of old node numbers to new node numbers

        Returns:
            new_node_number: it is -1 if the node number is not in the mapping table

        """
        try:
            new_node_number = mapping[old_node_number]
        except KeyError:
            new_node_number = -1

        return new_node_number

    def to_vtk(self, output_folder: str = None, alpha: float = 1.0):
        """Returns a vtk file of the DIC data.

        Args:
            output_folder: path to output folder; If None, no vtk file will be saved. The output file name will be the
                         same as the input file name with the extension .vtk
            alpha: alpha value for the Delaunay triangulation (only used if no connectivity data is provided)

        Returns:
            PyVista mesh object
        """
        # create node
        nodes = np.stack((self.coor_x, self.coor_y, self.coor_z), axis=1)

        # remap node numbers
        old_facet_id = self.facet_id - 1
        new_facet_id = np.arange(0, len(self.facet_id))
        mapping = dict(zip(old_facet_id, new_facet_id))
        renumber_nodes_vec = np.vectorize(self._renumber_nodes)

        # create mesh
        if self.connections is not None:
            # define elements
            elements = self.connections[:, 1:5]
            elements[:, 0] = 3
            elements[:, 1:] = renumber_nodes_vec(elements[:, 1:], mapping)

            # Check if there are any elements with -1 as node number
            mask = np.any(elements[:, 1:] == -1, axis=1)
            elements = elements[~mask]

            # define cell types
            cell_types = np.full(len(elements[:, 0]), fill_value=CellType.TRIANGLE, dtype=np.uint8)

            # define mesh
            mesh = pyvista.UnstructuredGrid(elements, cell_types, nodes)

        else:
            print(f'No connectivity data provided for {self.nodemap_name}. Reconstructing a mesh from the nodes using Delaunay triangulation with alpha = {alpha}.')
            cloud = pyvista.wrap(nodes)
            mesh = cloud.delaunay_2d(alpha=alpha)


        # add data
        mesh.point_data['x [mm]'] = self.coor_x
        mesh.point_data['y [mm]'] = self.coor_y
        mesh.point_data['z [mm]'] = self.coor_z
        mesh.point_data['u_x [mm]'] = self.disp_x
        mesh.point_data['u_y [mm]'] = self.disp_y
        mesh.point_data['u_z [mm]'] = self.disp_z
        mesh.point_data['u_sum [mm]'] = np.sqrt(self.disp_x ** 2 + self.disp_y ** 2 + self.disp_z ** 2)
        mesh.point_data['eps_x [%]'] = self.eps_x * 100.0
        mesh.point_data['eps_y [%]'] = self.eps_y * 100.0
        mesh.point_data['eps_xy [1]'] = self.eps_xy
        mesh.point_data['eps_vm [%]'] = self.eps_vm * 100.0
        mesh.point_data['sig_x [MPa]'] = self.sig_x
        mesh.point_data['sig_y [MPa]'] = self.sig_y
        mesh.point_data['sig_xy [MPa]'] = self.sig_xy
        mesh.point_data['sig_vm [MPa]'] = self.sig_vm

        # Compute the principal stresses
        self._calculate_principal_stresses()
        mesh.point_data['sig_1 [MPa]'] = self.sig_1
        mesh.point_data['sig_2 [MPa]'] = self.sig_2

        # Compute the principal strains
        self._calculate_principal_strains()
        mesh.point_data['eps_1 [%]'] = self.eps_1 * 100.0
        mesh.point_data['eps_2 [%]'] = self.eps_2 * 100.0

        # add metadata
        self.read_header()
        for meta_attribute in self.meta_attributes:
            meta_data = getattr(self, meta_attribute)
            mesh.field_data[meta_attribute] = [meta_data]

        # write vtk file
        if output_folder is not None:
            path = os.path.join(output_folder, self.nodemap_name[:-4] + '.vtk')
            mesh.save(path, binary=False)
        return mesh

    @staticmethod
    def _cut_nans(df):
        """Reads an array and deletes each row containing any nan value."""
        cut_nans_array = df[~np.isnan(df).any(axis=1)]
        return cut_nans_array

    @staticmethod
    def _cut_none_elements(df):
        """Reads an array and deletes each row containing any '-1' value."""
        mask = np.any(df == -1, axis=1)
        cut_none_elements = df[~mask]
        return cut_none_elements


def apply_mask(data: InputData, mask: np.array) -> InputData:
    masked_data = InputData()
    # apply mask to data
    masked_data.coor_x = data.coor_x[mask]
    masked_data.coor_y = data.coor_y[mask]
    masked_data.disp_x = data.disp_x[mask]
    masked_data.disp_y = data.disp_y[mask]
    masked_data.eps_x = data.eps_x[mask]
    masked_data.eps_y = data.eps_y[mask]
    masked_data.eps_xy = data.eps_xy[mask]
    masked_data.eps_vm = data.eps_vm[mask]
    masked_data.sig_x = data.sig_x[mask]
    masked_data.sig_y = data.sig_y[mask]
    masked_data.sig_xy = data.sig_xy[mask]
    masked_data.sig_vm = data.sig_vm[mask]
    return masked_data


class CrackTipInfo:
    """Wrapper for crack tip information.

    Methods:
        * set_manually - manually redefine crack tip information

    """

    def __init__(
            self,
            crack_tip_x: float = None,
            crack_tip_y: float = None,
            crack_tip_angle: float = None,
            left_or_right: str = None
    ):
        """Initialize crack tip info with provided attributes.

        Args:
            crack_tip_x: x-coordinate of the actual crack tip
            crack_tip_y: y-coordinate of the actual crack tip
            crack_tip_angle: angle of crack path between 0 and 180 degree
            left_or_right: either 'l' or 'r'

        """
        self.crack_tip_x = crack_tip_x
        self.crack_tip_y = crack_tip_y
        self.crack_tip_angle = crack_tip_angle
        self.left_or_right = left_or_right

    def set_manually(self, crack_tip_x: float, crack_tip_y: float, crack_tip_angle: float, left_or_right: str):
        """Alternatively coordinates may be given externally, e.g. from the crack detection module.

        Args:
            crack_tip_x: x-coordinate of the actual crack tip
            crack_tip_y: y-coordinate of the actual crack tip
            crack_tip_angle: angle of crack path between 0 and 180 degree
            left_or_right: either 'l' or 'r'

        """
        self.crack_tip_x = crack_tip_x
        self.crack_tip_y = crack_tip_y
        self.crack_tip_angle = crack_tip_angle
        self.left_or_right = left_or_right
