import logging
from copy import deepcopy

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import label

from crackpy.fracture_analysis.crack_tip import get_crack_nearfield, eigenfunction, get_zhao_solutions
from crackpy.fracture_analysis.utils import ReusableLinearInterpolator
from crackpy.input.input_data import InputData, apply_mask
from crackpy.structure_elements.material import Material

logger = logging.getLogger(__name__)

DEFAULT_BUCKNER_CHEN_TERMS = [1, 2, 3, 4, 5]


class IntegralProperties:
    """Integral properties which are used for more than one line integration within Fracture Analysis.

    Methods:
        * set_automatically - defines integral properties automatically using the input *data*

    """

    def __init__(
            self,
            number_of_paths: int = 9,
            integral_tick_size: float = None,
            number_of_nodes: int = None,

            integral_size_left: float = None,
            integral_size_right: float = None,
            integral_size_bottom: float = None,
            integral_size_top: float = None,

            top_offset: float = None,
            bottom_offset: float = None,

            paths_distance_left: float = None,
            paths_distance_right: float = None,
            paths_distance_top: float = None,
            paths_distance_bottom: float = None,

            mask_tolerance: float = None,

            buckner_williams_terms: list = None
    ):
        """Initialize integral path properties.

        Args:
            number_of_paths: number of integration paths for one crack tip (usually should be >=9)
            integral_tick_size: distance between integration points (i.e. 0.5 mm). If None, then number_of_nodes
                                is used instead, or the tick_size needs to be calculated from the pipeline
                                method *find_integral_props*
            number_of_nodes: number of integral nodes per path. If None, number of nodes varies for each path and
                                   is calculated from the integral_tick_size.
            integral_size_left: size of first integration path from crack tip to left boarder
                                (use negative value)
            integral_size_right: size of first integration path from crack tip to right boarder
            integral_size_bottom: size of first integration path from crack tip to bottom
                                  (use negative value)
            integral_size_top: size of first integration from crack tip to top
            top_offset: distance of unclosed part of the integral from the crack path (top side)
            bottom_offset: distance of unclosed part of the integral from the crack path (bottom side)
                            (use negative value)
            paths_distance_left: distance of integration paths (left side)
            paths_distance_right: distance of integration paths (right side)
            paths_distance_top: distance of integration paths (top side)
            paths_distance_bottom: distance of integration paths (bottom side)
            mask_tolerance: tolerance of the quadratic interpolation mask around the integration path
                            (fails if too small)

        """
        self.number_of_paths = number_of_paths
        self.integral_tick_size = integral_tick_size
        self.number_of_nodes = number_of_nodes

        self.integral_size_left = integral_size_left
        self.integral_size_right = integral_size_right
        self.integral_size_bottom = integral_size_bottom
        self.integral_size_top = integral_size_top

        self.top_offset = top_offset
        self.bottom_offset = bottom_offset

        self.paths_distance_left = paths_distance_left
        self.paths_distance_right = paths_distance_right
        self.paths_distance_top = paths_distance_top
        self.paths_distance_bottom = paths_distance_bottom

        self.mask_tolerance = mask_tolerance

        self.buckner_williams_terms = buckner_williams_terms

    def set_automatically(self, data: InputData, auto_detect_threshold: float):
        """Automatically set up the integration path properties.

        Args:
            data: obj of class InputData, used for auto-detection
            auto_detect_threshold: threshold stress typically taken equal to yield stress

        """
        logger.debug("Starting automatic integral path detection with threshold=%.2f MPa", auto_detect_threshold)

        if data.sig_vm is None:
            raise ValueError("Stresses need to be calculated before using ``data`` by calling data.calc_stresses()")
        # Calculate face size
        facet_size = data.get_facet_size()
        logger.debug("Calculated facet size: %.4f mm", facet_size)

        # Map data on regular grid
        x_min = facet_size * 2.0
        grid_x, grid_y = np.mgrid[-x_min:max(data.coor_x): 500j, min(data.coor_y):max(data.coor_y): 500j]
        ngrid = griddata((data.coor_x, data.coor_y), data.sig_vm, (grid_x, grid_y), method='linear')

        # Apply threshold
        threshold_array = ngrid > auto_detect_threshold
        threshold_array = threshold_array.astype(int)
        labeled_images, num_features = label(threshold_array)
        logger.debug("Found %d features above threshold", num_features)

        object_label = -1

        for i_feature in range(1, num_features + 1):
            mask = labeled_images == i_feature
            if np.any(np.all([grid_x[mask] > -facet_size,
                              grid_x[mask] < facet_size,
                              grid_y[mask] > -facet_size,
                              grid_y[mask] < facet_size], axis=0)):
                object_label = i_feature

        labeled_image = (labeled_images == object_label).astype(int)
        dist_factor = 2.0
        object_indices = np.argwhere(labeled_image > 0)

        if self.integral_size_right is None:
            self.integral_size_right = grid_x[max(object_indices[:, 0]), 0] + dist_factor * facet_size
        if self.integral_size_left is None:
            self.integral_size_left = - self.integral_size_right * 0.5 - facet_size
        if self.integral_size_bottom is None:
            self.integral_size_bottom = grid_y[0, min(object_indices[:, 1])] - dist_factor * facet_size
        if self.integral_size_top is None:
            self.integral_size_top = grid_y[0, max(object_indices[:, 1])] + dist_factor * facet_size
        if self.integral_tick_size is None:
            self.integral_tick_size = facet_size / 4.0

        # Set offsets
        ticks = int((self.integral_size_top - self.integral_size_bottom) / self.integral_tick_size * 8.0)
        coor_y = np.linspace(self.integral_size_bottom, self.integral_size_top, ticks, endpoint=True)
        coor_x = coor_y * 0.0 + self.integral_size_left
        sigma_vm_interpolated = griddata((data.coor_x, data.coor_y),
                                         data.sig_vm,
                                         (coor_x, coor_y),
                                         method='linear')

        y_min = 10000.0
        y_max = -10000.0
        for i, stress in enumerate(sigma_vm_interpolated):
            if abs(stress) > auto_detect_threshold:
                if coor_y[i] < y_min:
                    y_min = coor_y[i]
                if coor_y[i] > y_max:
                    y_max = coor_y[i]
        if y_min == 10000.0:
            y_min = 0
        if y_max == -10000.0:
            y_max = 0

        if self.top_offset is None:
            self.top_offset = y_max
        if self.bottom_offset is None:
            self.bottom_offset = y_min

        # Set path distances
        if self.paths_distance_left is None:
            self.paths_distance_left = facet_size
        if self.paths_distance_right is None:
            self.paths_distance_right = facet_size
        if self.paths_distance_top is None:
            self.paths_distance_top = facet_size
        if self.paths_distance_bottom is None:
            self.paths_distance_bottom = facet_size

        logger.debug("Automatic integral path detection completed:")
        logger.debug("  Integral sizes: left=%.2f, right=%.2f, top=%.2f, bottom=%.2f",
                     self.integral_size_left, self.integral_size_right, self.integral_size_top,
                     self.integral_size_bottom)
        logger.debug("  Offsets: top=%.2f, bottom=%.2f", self.top_offset, self.bottom_offset)
        logger.debug("  Tick size: %.4f mm", self.integral_tick_size)


class PathProperties:
    def __init__(self, size_left: float, size_right: float, size_bottom: float, size_top: float, tick_size: float,
                 num_nodes: int, top_offset: float, bottom_offset: float):
        """Properties of one single line integration path.

        Args:
            size_left: size of integration path from crack tip to left boarder (use negative value)
            size_right: size of integration path from crack tip to right boarder
            size_bottom: size of integration path from crack tip to bottom (use negative value)
            size_top: size of integration from crack tip to top
            tick_size: distance between integration points (i.e. 0.5 mm)
            num_nodes: number of nodes per integration path
            top_offset: distance of unclosed part of the integral from the crack path (top side)
            bottom_offset: distance of unclosed part of the integral from the crack path (top side)

        """
        self.size_left = size_left
        self.size_right = size_right
        self.size_bottom = size_bottom
        self.size_top = size_top
        self.tick_size = tick_size
        self.number_of_nodes = num_nodes
        self.top_offset = top_offset
        self.bottom_offset = bottom_offset


class IntegrationPath:
    """Wrapper for integration path functionalities.

    Methods:
        * create_nodes - list nodes specifying elements of the integration path
        * get_integration_points - coordinates of integration points and element sizes

    """

    def __init__(self, origin_x: float = 0.0, origin_y: float = 0.0, path_properties: PathProperties = None):
        """Initialize path properties.

        Args:
            origin_x: (float) refers to the point used as center for the integral.
            origin_y: (float) refers to the point used as center for the integral.
            path_properties: (PathProperties object) considering the base integration path plot_sets for size
                                                     and shape of the integration path
        
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.path_properties = path_properties

        self.nodes = None
        self.int_points = None

        self._create_nodes()
        self._create_integration_points()

    def _create_nodes(self):
        """Creates a list of nodes specifying the elements for the integration path. The number of nodes is taken from
        self.path_properties and then constant for each path or it is calculated from self.path_properties.tick_size and
        then it varies for each path due to constant tick_size. If neither tick_size nor num_of_nodes is None, then
        num_of_nodes is taken and tick_size is ignored.

        If only path.number_of_nodes is given, then the tick_size is calculated and written into self.path_properties.

        Returns:
            int_points with all x-coors at index 0 and all y-coors at index 1, again as lists

        """
        # restore integral_properties as method variable
        path = self.path_properties

        nodes_x_coors = []
        nodes_y_coors = []

        # section lengths
        length_bottom_left = path.bottom_offset - path.size_bottom
        length_bottom = path.size_right - path.size_left
        length_right = path.size_top - path.size_bottom
        length_top = length_bottom
        length_top_left = path.size_top - path.top_offset
        # whole length of integration path
        length_sum = length_bottom_left + length_bottom + length_right + length_top + length_top_left

        # Set number of edges
        if path.number_of_nodes is None:
            if path.tick_size is not None:
                self.path_properties.number_of_nodes = int(length_sum / path.tick_size) + 1
            else:
                raise ValueError("Either number of edges or integral tick size needs to be specified!")
        else:
            # calculate tick size from number of nodes!
            self.path_properties.tick_size = length_sum / path.number_of_nodes
        number_of_nodes = self.path_properties.number_of_nodes

        # Definition of integration points
        # bottom left
        num_nodes = int(length_bottom_left / length_sum * number_of_nodes)
        if num_nodes == 0:
            raise ValueError("Number of nodes is zero. Choose a smaller integral tick size!")
        for i in range(num_nodes + 1):
            nodes_x_coors.append(self.origin_x + path.size_left)
            nodes_y_coors.append(self.origin_y + path.bottom_offset - length_bottom_left / num_nodes * i)
        # bottom
        num_nodes = int(length_bottom / length_sum * number_of_nodes)
        for i in range(1, num_nodes + 1):
            nodes_x_coors.append(self.origin_x + path.size_left + length_bottom / num_nodes * i)
            nodes_y_coors.append(self.origin_y + path.size_bottom)
        # right
        num_nodes = int(length_right / length_sum * number_of_nodes)
        for i in range(1, num_nodes + 1):
            nodes_x_coors.append(self.origin_x + path.size_right)
            nodes_y_coors.append(self.origin_y + path.size_bottom + length_right / num_nodes * i)
        # top
        num_nodes = int(length_top / length_sum * number_of_nodes)
        for i in range(1, num_nodes + 1):
            nodes_x_coors.append(self.origin_x + path.size_right - length_top / num_nodes * i)
            nodes_y_coors.append(self.origin_y + path.size_top)
        # top left
        num_nodes = int(length_top_left / length_sum * number_of_nodes)
        for i in range(1, num_nodes + 1):
            nodes_x_coors.append(self.origin_x + path.size_left)
            nodes_y_coors.append(self.origin_y + path.size_top - length_top_left / num_nodes * i)

        self.nodes = [nodes_x_coors, nodes_y_coors]

    def _create_integration_points(self):
        """Generates coordinates of integration points and element sizes.

        Returns:
            int_points [[x_coor, y_coor, delta_x, delta_y]]

        """
        nodes_coors_x = self.nodes[0]
        nodes_coors_y = self.nodes[1]
        int_points = []
        for i in range(len(nodes_coors_x) - 1):
            x_coor = nodes_coors_x[i] + (nodes_coors_x[i + 1] - nodes_coors_x[i]) / 2.0
            y_coor = nodes_coors_y[i] + (nodes_coors_y[i + 1] - nodes_coors_y[i]) / 2.0
            delta_x = nodes_coors_x[i + 1] - nodes_coors_x[i]
            delta_y = nodes_coors_y[i + 1] - nodes_coors_y[i]
            int_points.append([x_coor, y_coor, delta_x, delta_y])
        int_points = np.asarray(int_points)
        self.int_points = int_points

    def get_integration_points(self) -> np.ndarray:
        """Get integration points of the integration path.

        Returns:
            int_points [[x_coor, y_coor, delta_x, delta_y]]

        """
        return self.int_points


class LineIntegral:
    """Line integral object for solving J-Integral and Interaction Integral for given material's input data
    and path of integration.

    Methods:
        * integrate - solve for J-integral, interaction integral, Bueckner/Chen integral

    """

    def __init__(self,
                 integration_path: IntegrationPath,
                 data: InputData,
                 material: Material,
                 mask_tol: float = None,
                 buckner_williams_terms: list = None):
        """Get integration points and interpolate data onto grid.

        Args:
            integration_path: obj of class IntegrationPath
            data: obj of class InputData, Input data object containing the full field DIC or FE data
            material: obj of class Material
            mask_tol: (float or None) tolerance of the quadratic interpolation mask around the integration path
            buckner_williams_terms: (list or None) list of Williams coefficients to be calculated with Buckner-Chen Integral

        """
        # input
        self.data = data
        self.data_orig = None  # for mode decomposition
        self.integration_path = integration_path
        self.x_shift = integration_path.path_properties.tick_size
        self.origin_x = integration_path.origin_x
        self.origin_y = integration_path.origin_y
        self.material = material
        self.buckner_williams_terms = buckner_williams_terms
        self.mask_tol = mask_tol
        self.np_integration_points = None
        self.tri = None

        # output
        self.j_integral = None
        self.decomp_j_integral_I = None
        self.decomp_j_integral_II = None
        self.decomp_j_integral_III = None
        self.decomp_j_integral_K_I = None
        self.decomp_j_integral_K_II = None
        self.decomp_j_integral_K_III = None
        self.sif_k_j = None
        self.sif_k_i = None
        self.sif_k_ii = None
        self.t_stress_sdm = None
        self.t_stress_chen = None
        self.t_stress_int = None
        self.williams_a_n = []
        self.williams_b_n = []
        self.williams_coefficients = []

        self.__post_init__()

    def __post_init__(self):
        self.np_integration_points = self.integration_path.get_integration_points()
        self._interpolate_on_integration_points()

    def integrate_all(self):
        """Call this method to solve all integrals
        - J-integral
        - Mode I, II, III decomposition of J-integral
        - Interaction integral for SIF K_I and K_II
        - T-stress with interaction integral method
        - T-stress with stress difference method
        - Williams coefficients with Buckner-Chen method (if terms are given)

        """
        logger.debug("Starting integration for all methods, integration points: %d", len(self.np_integration_points))

        self.integrate_j()
        logger.debug("J-integral: %.6f N/mm, K_J: %.6f MPa√m", self.j_integral, self.sif_k_j)

        self.integrate_j_decompose()
        logger.debug("J-decomposition completed: K_I=%.6f, K_II=%.6f, K_III=%.6f MPa√m",
                     self.decomp_j_integral_K_I, self.decomp_j_integral_K_II, self.decomp_j_integral_K_III)

        self.integrate_i_k1_k2()
        logger.debug("Interaction integral SIFs: K_I=%.6f, K_II=%.6f MPa√m", self.sif_k_i, self.sif_k_ii)

        self.integrate_i_t()
        logger.debug("T-stress (interaction): %.6f MPa", self.t_stress_int)

        self.integrate_t_sdm()
        logger.debug("T-stress (SDM): %.6f MPa", self.t_stress_sdm)

        if self.buckner_williams_terms is not None:
            self.integrate_buckner_chen()
            logger.debug("Buckner-Chen integral completed for %d terms", len(self.buckner_williams_terms))

    ###########################################
    # METHODS FOR CALCULATING THE DESCRIPTORS #
    ###########################################

    def integrate_j(self):
        """Call this method to solve integrals for J-integral :math:`J`
        """
        self.j_integral = self._solve_j_integral()  # in N/mm
        self.sif_k_j = np.sqrt(np.abs(self.j_integral) / 1000.0 * self.material.E)  # MPa*sqrt(mm) -> MPa*sqrt(m)

    def integrate_j_decompose(self):
        """Call this method to solve integrals for J-integral :math:`J` and its mode I, II, III decomposition
            Negative J values are sanitized to NaN for SIF calculation.

        """
        #############################################
        # Mode decomposition of J integral
        # see: Molteno, M. R., & Becker, T. H. (2015). Mode I-III decomposition of the j-integral from DIC
        # displacement data. Strain, 51(6), 492–503. https://doi.org/10.1111/str.12166
        #############################################
        data_orig = deepcopy(self.data)
        self._map_displacement_data_on_regular_grid(grid_points=200)

        # Mode I
        self.data = self._prepare_mode_data(mode='I')
        self._interpolate_on_integration_points()
        self.decomp_j_integral_I = self._solve_j_integral()  # in N/mm
        j_san = np.where(self.decomp_j_integral_I >= 0, self.decomp_j_integral_I, np.nan)
        self.decomp_j_integral_K_I = np.sqrt(j_san * self.material.E) / np.sqrt(1000)  # MPa*sqrt(m)

        # Mode II
        self.data = self._prepare_mode_data(mode='II')
        self._interpolate_on_integration_points()
        self.decomp_j_integral_II = self._solve_j_integral()  # in N/mm
        j_san = np.where(self.decomp_j_integral_II >= 0, self.decomp_j_integral_II, np.nan)
        self.decomp_j_integral_K_II = np.sqrt(j_san * self.material.E) / np.sqrt(1000)  # MPa*sqrt(m)

        # Mode III
        self.data = self._prepare_mode_data(mode='III')
        self._interpolate_on_integration_points()
        self._interpolate_on_integration_points_z()
        self.decomp_j_integral_III = self._solve_j_integral_III()  # in N/mm
        j_san = np.where(self.decomp_j_integral_III >= 0, self.decomp_j_integral_III, np.nan)
        self.decomp_j_integral_K_III = np.sqrt(j_san * self.material.E /
                                               (1 + self.material.nu_xy)) / np.sqrt(1000)  # MPa*sqrt(m)

        # Restore original data
        self.data = data_orig
        self._interpolate_on_integration_points()

    def integrate_i_k1_k2(self):
        #############################################
        # see Meinhard Kuna Section 6.7.2 for details
        #############################################
        # Mode I auxiliary loading condition
        ki_aux = 1
        kii_aux = 0
        ki_aux = self._unit_m_to_mm(ki_aux)
        kii_aux = self._unit_m_to_mm(kii_aux)
        interaction_integral_value = self._solve_interaction_integral(ki_aux, kii_aux)
        self.sif_k_i = self.material.E / ki_aux * interaction_integral_value / 2  # mistake in Kuna formula (6.86)
        self.sif_k_i = self._unit_mm_to_m(self.sif_k_i)  # MPa*sqrt(mm) -> MPa*sqrt(m)

        # Mode II auxiliary loading condition
        ki_aux = 0
        kii_aux = 1
        ki_aux = self._unit_m_to_mm(ki_aux)
        kii_aux = self._unit_m_to_mm(kii_aux)
        interaction_integral_value = self._solve_interaction_integral(ki_aux, kii_aux)
        self.sif_k_ii = self.material.E / kii_aux * interaction_integral_value / 2  # mistake in Kuna formula (6.86)
        self.sif_k_ii = self._unit_mm_to_m(self.sif_k_ii)  # MPa*sqrt(mm) -> MPa*sqrt(m)

    def integrate_i_t(self):
        # T-stress with interaction integral method
        int_path_max_x = np.max(self.np_integration_points[:, 0])
        t_stress_integral = self._solve_t_stress_interaction_integral()
        if self.material.plane_strain:
            # plane strain
            self.t_stress_int = self.material.E / (1 - self.material.nu_xy ** 2) * t_stress_integral
        else:
            eps_z = griddata((self.data.coor_x, self.data.coor_y),
                             - self.material.nu_xy * (self.data.eps_x + self.data.eps_y),
                             (int_path_max_x, 0)).item()
            # plane stress
            self.t_stress_int = self.material.E * (t_stress_integral + self.material.nu_xy * eps_z)

    def integrate_t_sdm(self):
        # T-stress with stress difference method
        int_path_max_x = np.max(self.np_integration_points[:, 0])
        self.t_stress_sdm = griddata((self.data.coor_x, self.data.coor_y),
                                     self.data.sig_x - self.data.sig_y,
                                     (int_path_max_x, 0))

    def integrate_buckner_chen(self):
        # Williams coefficents with Chen method
        for n in self.buckner_williams_terms:
            a_n = self._williams_coeff_from_chen_integral(a_aux=1, n=n)
            b_n = self._williams_coeff_from_chen_integral(b_aux=1, n=n)
            self.williams_a_n.append(a_n)
            self.williams_b_n.append(b_n)
            self.williams_coefficients.append([n, a_n, b_n])

        # T-stress with Bueckner / Chen integral method
        n = 2
        c_m = 1
        a_n = self._williams_coeff_from_chen_integral(a_aux=c_m, n=n)
        self.t_stress_chen = 4 * a_n  # MPa

    #####################################
    # METHODS FOR SOLVING THE INTEGRALS #
    #####################################

    # base method for J-integral, used in child classes
    def _solve_j_integral(self) -> float:
        """Function that returns the J-integral as a line integration.

        Returns:
            J-integral value

        """
        j_int_value = 0.0
        for i in range(len(self.np_integration_points[:, 0])):
            int_point_sig_tensor = np.asarray([[self.interpolated_sig_x[i], self.interpolated_sig_xy[i]],
                                               [self.interpolated_sig_xy[i], self.interpolated_sig_y[i]]])
            int_point_eps_tensor = np.asarray([[self.interpolated_eps_x[i], self.interpolated_eps_xy[i]],
                                               [self.interpolated_eps_xy[i], self.interpolated_eps_y[i]]])

            elem_height = self.np_integration_points[i, 3]
            elem_size = np.sqrt(self.np_integration_points[i, 2] ** 2.0 + self.np_integration_points[i, 3] ** 2.0)
            direction_vector = [self.np_integration_points[i, 2], self.np_integration_points[i, 3], 0.0]
            normal_vector = np.cross(direction_vector, [0.0, 0.0, 1.0])
            normal_vector = np.asarray(1.0 / np.linalg.norm(normal_vector) * normal_vector)[0:2]
            t_vector = np.dot(int_point_sig_tensor, normal_vector)

            energy_term = 0.5 * np.sum(int_point_sig_tensor * int_point_eps_tensor)
            stress_term = (t_vector[0] * self.interpolated_eps_x[i] + t_vector[1] * self.interpolated_disp_y_dx[i])
            int_point_j = energy_term * elem_height - stress_term * elem_size
            j_int_value += int_point_j  # if not np.isnan(int_point_j) else 0.0

        return j_int_value

    def _solve_j_integral_III(self) -> float:
        """Function that returns the J-integral as a line integration.

        Returns:
            J-integral value

        """
        j_int_value = 0.0
        for i in range(len(self.np_integration_points[:, 0])):
            elem_height = self.np_integration_points[i, 3]
            elem_size = np.sqrt(self.np_integration_points[i, 2] ** 2.0 + self.np_integration_points[i, 3] ** 2.0)
            direction_vector = [self.np_integration_points[i, 2], self.np_integration_points[i, 3], 0.0]
            normal_vector = np.cross(direction_vector, [0.0, 0.0, 1.0])
            normal_vector = np.asarray(1.0 / np.linalg.norm(normal_vector) * normal_vector)[0:2]

            energy_term = (self.interpolated_eps_xz[i] * self.interpolated_sigma_xz[i] +
                           self.interpolated_eps_yz[i] * self.interpolated_sigma_yz[i])
            stress_term = (self.interpolated_sigma_xz[i] * self.interpolated_eps_xz[i] * normal_vector[0] +
                           self.interpolated_sigma_yz[i] * self.interpolated_eps_xz[i] * normal_vector[1])

            int_point_j = energy_term * elem_height - stress_term * elem_size
            j_int_value += int_point_j
        return j_int_value

    def _solve_interaction_integral(self, ki_aux: float, kii_aux: float) -> float:
        """Function that calculates the interaction integral as a line integration.
        More precisely, this function returns :math:`J^{1,2}` from formula 6.81 in Meinhard Kuna's book.

        Args:
            ki_aux: usually 1.0 if K_I should be calculated and else 0.0
            kii_aux: usually 1.0 if K_II should be calculated and else 0.0

        Returns:
            interaction integral value

        """
        interaction_integral_value = 0.0

        for i in range(len(self.np_integration_points[:, 0])):
            # define stress and strain tensors
            int_point_sig_tensor = np.asarray([[self.interpolated_sig_x[i], self.interpolated_sig_xy[i]],
                                               [self.interpolated_sig_xy[i], self.interpolated_sig_y[i]]])

            # size and height of integration elements
            elem_height = self.np_integration_points[i, 3]
            elem_size = np.sqrt(self.np_integration_points[i, 2] ** 2.0 + self.np_integration_points[i, 3] ** 2.0)
            # normal vector
            direction_vector = [self.np_integration_points[i, 2], self.np_integration_points[i, 3], 0.0]
            normal_vector = np.cross(direction_vector, [0.0, 0.0, 1.0])
            normal_vector = np.asarray(1.0 / np.linalg.norm(normal_vector) * normal_vector)[0:2]

            # traction
            t_vector = np.dot(int_point_sig_tensor, normal_vector)

            # auxiliary stress, strain, and displacement fields
            r, phi = self._make_polar(x=-self.origin_x + self.np_integration_points[i, 0],
                                      y=-self.origin_y + self.np_integration_points[i, 1])
            crack_nearfield_sig, crack_nearfield_eps, _ = get_crack_nearfield(ki_aux, kii_aux, r, phi, self.material)
            # auxiliary traction
            t_vector_analytic = np.dot(crack_nearfield_sig, normal_vector)

            # calculate x-derivative of auxiliary y-displacement
            r, phi = self._make_polar(x=-self.origin_x + self.np_integration_points[i, 0] + self.x_shift,
                                      y=-self.origin_y + self.np_integration_points[i, 1])
            _, _, crack_nearfield_disp_shift_right = get_crack_nearfield(ki_aux, kii_aux, r, phi, self.material)

            r, phi = self._make_polar(x=-self.origin_x + self.np_integration_points[i, 0] - self.x_shift,
                                      y=-self.origin_y + self.np_integration_points[i, 1])
            _, _, crack_nearfield_disp_shift_left = get_crack_nearfield(ki_aux, kii_aux, r, phi, self.material)

            disp_y_dx_aux = (crack_nearfield_disp_shift_right[1] - crack_nearfield_disp_shift_left[1]) / \
                            (2 * self.x_shift)

            # Interaction energy integral
            energy_term_aux = np.sum(int_point_sig_tensor * crack_nearfield_eps)
            eps_x_aux = crack_nearfield_eps[0, 0]
            delta_int_1 = energy_term_aux * elem_height  # normal in x-direction is zero for elem_height = 0
            delta_int_2 = -(t_vector[0] * eps_x_aux + t_vector[1] * disp_y_dx_aux) * elem_size
            delta_int_3 = -(t_vector_analytic[0] * self.interpolated_eps_x[i]
                            + t_vector_analytic[1] * self.interpolated_disp_y_dx[i]) * elem_size

            interaction_integral_value += delta_int_1 + delta_int_2 + delta_int_3

        return interaction_integral_value

    def _solve_t_stress_interaction_integral(self) -> float:
        """Interaction path integral for the determination of T-stress according to Cardew et al. '85, Kfouri '86,
        Zhao et al. '01 and others. The analogous domain integral is used in ABACUS and ANSYS to determine T-stress.

        Returns:
            T stress interaction integral value

        """
        interaction_integral_value = 0.0

        for i in range(len(self.np_integration_points[:, 0])):
            # define stress and strain tensors
            int_point_eps_tensor = np.asarray([[self.interpolated_eps_x[i], self.interpolated_eps_xy[i]],
                                               [self.interpolated_eps_xy[i], self.interpolated_eps_y[i]]])
            int_point_sig_tensor = np.asarray([[self.interpolated_sig_x[i], self.interpolated_sig_xy[i]],
                                               [self.interpolated_sig_xy[i], self.interpolated_sig_y[i]]])

            # size and height of integration elements
            elem_height = self.np_integration_points[i, 3]
            elem_size = np.sqrt(self.np_integration_points[i, 2] ** 2.0 + self.np_integration_points[i, 3] ** 2.0)
            # normal vector
            direction_vector = [self.np_integration_points[i, 2], self.np_integration_points[i, 3], 0.0]
            normal_vector = np.cross(direction_vector, [0.0, 0.0, 1.0])
            normal_vector = np.asarray(1.0 / np.linalg.norm(normal_vector) * normal_vector)[0:2]

            # traction
            t_vector = np.dot(int_point_sig_tensor, normal_vector)

            # auxiliary stress, strain, and displacement fields (see Zhao et al. 2001)
            r, phi = self._make_polar(x=-self.origin_x + self.np_integration_points[i, 0],
                                      y=-self.origin_y + self.np_integration_points[i, 1])
            sigma_x_aux, sigma_y_aux, sigma_xy_aux, _, _ = get_zhao_solutions(r, phi, self.material)
            sigma_tensor_aux = np.asarray([[sigma_x_aux, sigma_xy_aux],
                                           [sigma_xy_aux, sigma_y_aux]])

            # auxiliary traction
            t_vector_aux = np.dot(sigma_tensor_aux, normal_vector)

            # calculate x-derivative of auxiliary y-displacement
            r, phi = self._make_polar(x=-self.origin_x + self.np_integration_points[i, 0] + self.x_shift,
                                      y=-self.origin_y + self.np_integration_points[i, 1])
            _, _, _, u_x_shift_right, u_y_shift_right = get_zhao_solutions(r, phi, self.material)

            r, phi = self._make_polar(x=-self.origin_x + self.np_integration_points[i, 0] - self.x_shift,
                                      y=-self.origin_y + self.np_integration_points[i, 1])
            _, _, _, u_x_shift_left, u_y_shift_left = get_zhao_solutions(r, phi, self.material)

            disp_x_dx_aux = (u_x_shift_right - u_x_shift_left) / (2 * self.x_shift)
            disp_y_dx_aux = (u_y_shift_right - u_y_shift_left) / (2 * self.x_shift)

            # Interaction energy integral
            energy_term_aux = np.sum(sigma_tensor_aux * int_point_eps_tensor)
            delta_int_1 = energy_term_aux * elem_height  # normal in x-direction is zero for elem_height = 0
            delta_int_2 = -(t_vector[0] * disp_x_dx_aux + t_vector[1] * disp_y_dx_aux) * elem_size
            delta_int_3 = -(t_vector_aux[0] * self.interpolated_eps_x[i]
                            + t_vector_aux[1] * self.interpolated_disp_y_dx[i]) * elem_size

            interaction_integral_value += delta_int_1 + delta_int_2 + delta_int_3

        return interaction_integral_value

    def _williams_coeff_from_chen_integral(self, a_aux=0, b_aux=0, n=1) -> float:
        """This method implements formula (6.94) from Meinhard Kuna's book on fracture mechanics.

        Args:
            a_aux: auxiliary state coefficient c_m
            b_aux: auxiliary state coefficient d_m
            n: order of Williams series coefficient

        Returns:
            the sought Williams coefficient of order n (length unit is mm !)

        """
        if a_aux != 0 and b_aux != 0:
            raise ValueError("Either a_aux or b_aux has to be zero!")
        integral_value = self._solve_chen_integral(n=-n, a_n=a_aux, b_n=b_aux)
        williams_coeff = - self.material.G / (self.material.kappa + 1) / (a_aux + b_aux) / \
                         (np.pi * n * (-1) ** (n + 1)) * integral_value
        return williams_coeff

    def _solve_chen_integral(self, n: int = -1, a_n: float = 0, b_n: float = 0) -> float:
        """Function that calculates the Bueckner / Chen integral as a line integration.
        [see Y. Z. Chen, New path independent integrals in linear elastic fracture mechanics, 1985]

        Args:
            n: order of corresponding Williams series coefficient to compute
            a_n: input coefficient to eigenfunction (typically either 0 or 1)
            b_n: input coefficient to eigenfunction (typically either 0 or 1)

        Returns:
            Chen integral value

        """
        chen_integral = 0.0
        for i in range(len(self.np_integration_points[:, 0])):
            # size and height of integration elements
            elem_size = np.sqrt(self.np_integration_points[i, 2] ** 2.0 + self.np_integration_points[i, 3] ** 2.0)

            # normal vector
            direction_vector = [self.np_integration_points[i, 2], self.np_integration_points[i, 3], 0.0]
            normal_vector = np.cross(direction_vector, [0.0, 0.0, 1.0])
            normal_vector = np.asarray(1.0 / np.linalg.norm(normal_vector) * normal_vector)[0:2]

            # stress and displacement of real data
            int_point_sig_tensor = np.asarray([[self.interpolated_sig_x[i], self.interpolated_sig_xy[i]],
                                               [self.interpolated_sig_xy[i], self.interpolated_sig_y[i]]])
            int_point_disp_vector = np.asarray([self.interpolated_disp_x[i], self.interpolated_disp_y[i]])
            # traction of real data
            t_vector = np.dot(int_point_sig_tensor, normal_vector)

            # auxiliary stress and displacement
            r, phi = self._make_polar(x=-self.origin_x + self.np_integration_points[i, 0],
                                      y=-self.origin_y + self.np_integration_points[i, 1])
            sigma_x_n, sigma_y_n, sigma_xy_n, disp_x_aux, disp_y_aux = eigenfunction(n, a_n, b_n, r, phi, self.material)
            sigma_aux = np.asarray([[sigma_x_n, sigma_xy_n], [sigma_xy_n, sigma_y_n]])
            disp_aux = np.asarray([disp_x_aux, disp_y_aux])
            # auxiliary traction
            t_vector_aux = np.dot(sigma_aux, normal_vector)

            # Chen energy integral
            chen_integral += np.sum(t_vector * disp_aux - t_vector_aux * int_point_disp_vector) * elem_size

        return chen_integral

    ######################################
    # FUNCTIONS FOR STRAIN RECALCULATION #
    ######################################

    def _compute_strains_xy(self, u_x, u_y, gap):
        steps = self.grid_points
        dist = self.x_coordinates[1] - self.x_coordinates[0]
        eps_xx = np.zeros_like(self.x_mesh)
        eps_xx[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(u_x[0:int(steps / 2) - gap, 0:int(steps / 2)],
                                                                       dist, axis=1)
        eps_xx[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(u_x[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                      dist, axis=1)
        eps_xx[:, int(steps / 2):] = np.gradient(u_x[:, int(steps / 2):], dist, axis=1)

        eps_yy = np.zeros_like(self.x_mesh)
        eps_yy[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(u_y[0:int(steps / 2) - gap, 0:int(steps / 2)],
                                                                       dist, axis=0)
        eps_yy[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(u_y[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                      dist, axis=0)
        eps_yy[:, int(steps / 2):] = np.gradient(u_y[:, int(steps / 2):], dist, axis=0)

        eps_xy = np.zeros_like(self.x_mesh)
        eps_xy[0:int(steps / 2) - gap, 0:int(steps / 2)] = 0.5 * (
                np.gradient(u_x[0:int(steps / 2) - gap, 0:int(steps / 2)], dist, axis=0) +
                np.gradient(u_y[0:int(steps / 2) - gap, 0:int(steps / 2)], dist, axis=1))
        eps_xy[int(steps / 2) + gap:, 0:int(steps / 2)] = 0.5 * (
                np.gradient(u_x[int(steps / 2) + gap:, 0:int(steps / 2)], dist, axis=0) +
                np.gradient(u_y[int(steps / 2) + gap:, 0:int(steps / 2)], dist, axis=1))
        eps_xy[:, int(steps / 2):] = 0.5 * (np.gradient(u_x[:, int(steps / 2):], dist, axis=0) +
                                            np.gradient(u_y[:, int(steps / 2):], dist, axis=1))
        return eps_xx, eps_yy, eps_xy

    def _compute_stress_strain_z(self, u_z, gap):
        steps = self.grid_points
        dist = self.x_coordinates[1] - self.x_coordinates[0]
        eps_xz = np.zeros_like(self.x_mesh)
        eps_xz[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(u_z[0:int(steps / 2) - gap, 0:int(steps / 2)],
                                                                       dist, axis=1)
        eps_xz[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(u_z[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                      dist, axis=1)
        eps_xz[:, int(steps / 2):] = np.gradient(u_z[:, int(steps / 2):], dist, axis=1)

        eps_yz = np.zeros_like(self.x_mesh)
        eps_yz[0:int(steps / 2) - gap, 0:int(steps / 2)] = np.gradient(u_z[0:int(steps / 2) - gap, 0:int(steps / 2)],
                                                                       dist, axis=0)
        eps_yz[int(steps / 2) + gap:, 0:int(steps / 2)] = np.gradient(u_z[int(steps / 2) + gap:, 0:int(steps / 2)],
                                                                      dist, axis=0)
        eps_yz[:, int(steps / 2):] = np.gradient(u_z[:, int(steps / 2):], dist, axis=0)

        sigma_xz = self.material.G * eps_xz
        sigma_yz = self.material.G * eps_yz

        return eps_xz, eps_yz, sigma_xz, sigma_yz

    #########################################################
    # FUNCTIONS FOR PREPARING DATA FOR J-MODE DECOMPOSITION #
    #########################################################

    def _prepare_mode_data(self, mode: str):
        """Prepare `InputData` for mode 1, 2 or 3 decomposition (j-integral).

        Args:
            mode: 1 / 'I' -> Mode I, 2 / 'II' -> Mode II, 3 / 'III' -> Mode III

        Returns:
            InputData instance populated for requested mode.
        """

        if mode == 'I':
            u_x = 0.5 * (self.disp_u_mesh + np.flipud(self.disp_u_mesh))
            u_y = 0.5 * (self.disp_v_mesh - np.flipud(self.disp_v_mesh))
            u_z = np.zeros_like(u_x)
        elif mode == 'II':
            u_x = 0.5 * (self.disp_u_mesh - np.flipud(self.disp_u_mesh))
            u_y = 0.5 * (self.disp_v_mesh + np.flipud(self.disp_v_mesh))
            u_z = np.zeros_like(u_x)
        elif mode == 'III':
            u_z = 0.5 * (self.disp_w_mesh - np.flipud(self.disp_w_mesh))
            u_x = np.zeros_like(u_z)
            u_y = np.zeros_like(u_z)
        else:
            raise ValueError("Mode has to be 'I', 'II' or 'III'!")

        eps_xx, eps_yy, eps_xy = self._compute_strains_xy(u_x, u_y, 2)
        eps_xz, eps_yz, sigma_xz, sigma_yz = self._compute_stress_strain_z(u_z, 2)

        # populate the container
        decomp_data = InputData()
        decomp_data.coor_x = self.x_mesh.flatten()
        decomp_data.coor_y = self.y_mesh.flatten()
        decomp_data.disp_x = u_x.flatten()
        decomp_data.disp_y = u_y.flatten()
        decomp_data.disp_z = u_z.flatten()
        decomp_data.eps_x = eps_xx.flatten()
        decomp_data.eps_y = eps_yy.flatten()
        decomp_data.eps_xy = eps_xy.flatten()
        decomp_data.eps_xz = eps_xz.flatten()
        decomp_data.eps_yz = eps_yz.flatten()
        decomp_data.sigma_xz = sigma_xz.flatten()
        decomp_data.sigma_yz = sigma_yz.flatten()

        # recalc Von Mises strains and stresses
        decomp_data.calc_eps_vm()
        decomp_data.calc_stresses(self.material)

        return decomp_data

    ################################
    # HELPER FUNCTIONS FOR MESHING #
    ################################

    def _map_displacement_data_on_regular_grid(self, grid_points: int):
        left = np.min(self.np_integration_points[:, 0])
        right = np.max(self.np_integration_points[:, 0])
        bottom = np.min(self.np_integration_points[:, 1])
        top = np.max(self.np_integration_points[:, 1])

        grid_size_x = max(abs(left), abs(right)) * 1.2
        grid_size_y = max(abs(bottom), abs(top)) * 1.2

        self.grid_points = int(grid_points)
        self.x_coordinates = np.linspace(-grid_size_x, grid_size_x, self.grid_points, endpoint=True)
        self.y_coordinates = np.linspace(-grid_size_y, grid_size_y, self.grid_points, endpoint=True)

        # used for j decomposition
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_coordinates, self.y_coordinates, indexing="xy")

        grid_points = np.c_[self.x_mesh.ravel(), self.y_mesh.ravel()]

        interp_disp = ReusableLinearInterpolator(self.data.coor_x, self.data.coor_y, grid_points)
        uvw = interp_disp.interpolate(np.c_[self.data.disp_x, self.data.disp_y, self.data.disp_z])
        uvw = uvw.reshape(self.x_mesh.shape + (3,))
        self.disp_u_mesh, self.disp_v_mesh, self.disp_w_mesh = uvw[:, :, 0], uvw[:, :, 1], uvw[:, :, 2]

    ###########################################
    # HELPER FUNCTIONS FOR DATA INTERPOLATION #
    ###########################################

    def _interpolate_on_integration_points(self):
        """Interpolates full field data onto the integration path coordinates.
        Further, calculates the interpolated results for shifted points for derivatives."""

        self.pos_shifted_np_int_points = np.asarray(self.np_integration_points[:, 0]) + self.x_shift
        self.neg_shifted_np_int_points = np.asarray(self.np_integration_points[:, 0]) - self.x_shift

        if self.mask_tol is not None:
            # mask out areas away from the integration points
            left = np.min(self.np_integration_points[:, 0])
            right = np.max(self.np_integration_points[:, 0])
            bottom = np.min(self.np_integration_points[:, 1])
            top = np.max(self.np_integration_points[:, 1])

            tol = self.mask_tol
            outer_square = (left - tol <= self.data.coor_x) * (self.data.coor_x <= right + tol) * \
                           (bottom - tol <= self.data.coor_y) * (self.data.coor_y <= top + tol)
            inner_square = (left + tol <= self.data.coor_x) * (self.data.coor_x <= right - tol) * \
                           (bottom + tol <= self.data.coor_y) * (self.data.coor_y <= top - tol)
            mask = np.where(outer_square * (1 - inner_square))  # outer_square \ inner_square
            data = apply_mask(self.data, mask)
        else:
            data = self.data

        # we reuses triangles and weights, since interpolation points are the same for all fields
        # for the fields
        interpolator = ReusableLinearInterpolator(data.coor_x, data.coor_y, np.c_[
            self.np_integration_points[:, 0], self.np_integration_points[:, 1]])
        intp_data = interpolator.interpolate(
            np.c_[data.eps_x, data.eps_y, data.eps_xy,
            data.sig_x, data.sig_y, data.sig_xy,
            data.disp_x, data.disp_y])
        (self.interpolated_eps_x, self.interpolated_eps_y, self.interpolated_eps_xy,
         self.interpolated_sig_x, self.interpolated_sig_y, self.interpolated_sig_xy,
         self.interpolated_disp_x, self.interpolated_disp_y) = (intp_data[:, 0], intp_data[:, 1], intp_data[:, 2],
                                                                intp_data[:, 3], intp_data[:, 4], intp_data[:, 5],
                                                                intp_data[:, 6], intp_data[:, 7])

        # for the derivative of disp_y w.r.t. x, we reuse the standard griddata approach (since points are different)
        self.interpolated_disp_y_dx_positive = griddata((data.coor_x, data.coor_y), data.disp_y,
                                                        (self.pos_shifted_np_int_points,
                                                         self.np_integration_points[:, 1]),
                                                        method='linear')
        self.interpolated_disp_y_dx_negative = griddata((data.coor_x, data.coor_y), data.disp_y,
                                                        (self.neg_shifted_np_int_points,
                                                         self.np_integration_points[:, 1]),
                                                        method='linear')
        self.interpolated_disp_y_dx = (self.interpolated_disp_y_dx_positive -
                                       self.interpolated_disp_y_dx_negative) / (2.0 * self.x_shift)

    def _interpolate_on_integration_points_z(self, mask_tol: float = None):
        """Interpolates full field data onto the integration path coordinates.
        Further, calculates the interpolated results for shifted points for derivatives.
        This method is used for the mode III decomposition of J integral.
        """

        self.pos_shifted_np_int_points = np.asarray(self.np_integration_points[:, 0]) + self.x_shift
        self.neg_shifted_np_int_points = np.asarray(self.np_integration_points[:, 0]) - self.x_shift

        if mask_tol is not None:
            # mask out areas away from the integration points
            left = np.min(self.np_integration_points[:, 0])
            right = np.max(self.np_integration_points[:, 0])
            bottom = np.min(self.np_integration_points[:, 1])
            top = np.max(self.np_integration_points[:, 1])

            tol = mask_tol
            outer_square = (left - tol <= self.data.coor_x) * (self.data.coor_x <= right + tol) * \
                           (bottom - tol <= self.data.coor_y) * (self.data.coor_y <= top + tol)
            inner_square = (left + tol <= self.data.coor_x) * (self.data.coor_x <= right - tol) * \
                           (bottom + tol <= self.data.coor_y) * (self.data.coor_y <= top - tol)
            mask = np.where(outer_square * (1 - inner_square))  # outer_square \ inner_square
            data = apply_mask(self.data, mask)
        else:
            data = self.data

        interpolator = ReusableLinearInterpolator(data.coor_x, data.coor_y, np.c_[
            self.np_integration_points[:, 0], self.np_integration_points[:, 1]])
        intp_data = interpolator.interpolate(np.c_[data.eps_xz, data.eps_yz, data.sigma_xz, data.sigma_yz])
        (self.interpolated_eps_xz, self.interpolated_eps_yz,
         self.interpolated_sigma_xz, self.interpolated_sigma_yz) = (intp_data[:, 0], intp_data[:, 1],
                                                                    intp_data[:, 2], intp_data[:, 3])

    ########
    # MISC #
    ########

    @staticmethod
    def _unit_m_to_mm(quantity_in_m, n=1):
        return quantity_in_m * 1000 ** (1 - n / 2)

    @staticmethod
    def _unit_mm_to_m(quantity_in_mm, n=1):
        return quantity_in_mm / 1000 ** (1 - n / 2)

    @staticmethod
    def _make_polar(x: float, y: float):
        """Takes cartesian coordinates and maps onto polar coordinates."""
        r = np.sqrt(x ** 2.0 + y ** 2.0)
        phi = np.arctan2(y, x)
        return r, phi

    @staticmethod
    def ensure_defaults_buckner_chen(options: IntegralProperties):
        """Ensures that the default terms for the Buckner-Chen Integral are provided."""
        if options.buckner_williams_terms is None:
            options.buckner_williams_terms = [1, 2, 3, 4, 5]
        elif 1 not in options.buckner_williams_terms:
            options.buckner_williams_terms.append(1)
            logger.info('Buckner-Williams terms should include 1. Term added.')
        if 0 in options.buckner_williams_terms:
            options.buckner_williams_terms.remove(0)
            logger.warning('Buckner-Williams terms should not include 0. Term removed.')
        options.buckner_williams_terms.sort()
