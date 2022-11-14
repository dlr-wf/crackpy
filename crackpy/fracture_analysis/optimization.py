import numpy as np
from scipy.interpolate import griddata
from scipy import optimize

from crackpy.fracture_analysis.crack_tip import williams_stress_field, williams_displ_field, cjp_displ_field
from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.material import Material


class OptimizationProperties:
    """Class for setting the Optimization properties."""

    def __init__(
            self,
            angle_gap: float or None = 20,
            min_radius: float or None = None,
            max_radius: float or None = None,
            tick_size: float or None = 0.01,
            terms=None
    ):
        """Initialize Optimization properties.

        Args:
            angle_gap: Angle gap between crack path and fitting domain.
                If None, angle_gap is set to 20.
            min_radius: minimum radius of fitting domain.
                If None, min_radius is set to crack_length / 20.
            max_radius: maximum radius of fitting domain.
                If None, max_radius is set to crack_length / 5.
            tick_size: tick size of fitting domain.
                If None, tick_size is set to 0.01.
            terms: (list or None) list of Williams terms to be used in optimization, e.g. [-1, 1, 2].

        """
        self.angle_gap = angle_gap
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.tick_size = tick_size
        self.terms = terms


class Optimization:
    """Optimization class

    Methods:
        * optimize_cjp_displacements
        * optimize_williams_displacements
        * optimize_williams_stresses

        * residuals_cjp_displacements
        * residuals_williams_displacements
        * residuals_williams_stresses

        * mse_williams_displacements
        * mse_williams_stresses

    """

    def __init__(self,
                 data: InputData,
                 material: Material = Material(),
                 options: OptimizationProperties = OptimizationProperties()):
        """Initializes Optimization arguments.

        Args:
            data: obj of class InputData
            material: obj of class Material
            options: obj of class OptimizationProperties

        """
        self.data = data
        self.material = material

        # polar grid & corresponding cartesian grid
        angle_gap = options.angle_gap
        self.min_radius = options.min_radius
        max_radius = options.max_radius
        tick_size = options.tick_size
        self.terms = np.asarray(options.terms)
        self.angle_gap_rad = angle_gap / 180 * np.pi
        self.r_grid, self.phi_grid = \
            np.mgrid[self.min_radius:max_radius:tick_size, -np.pi+self.angle_gap_rad:np.pi-self.angle_gap_rad:tick_size]
        self.x_grid, self.y_grid = self.make_cartesian(self.r_grid, self.phi_grid)

        # map transformed data to cartesian grid
        disp_x_0_0 = griddata((self.data.coor_x, self.data.coor_y), self.data.disp_x, (0, 0)).item()
        disp_y_0_0 = griddata((self.data.coor_x, self.data.coor_y), self.data.disp_y, (0, 0)).item()
        self.interp_sigma_x = griddata(points=(self.data.coor_x, self.data.coor_y),
                                       values=self.data.sig_x,
                                       xi=(self.x_grid, self.y_grid),
                                       method='linear')
        self.interp_sigma_y = griddata(points=(self.data.coor_x, self.data.coor_y),
                                       values=self.data.sig_y,
                                       xi=(self.x_grid, self.y_grid),
                                       method='linear')
        self.interp_sigma_xy = griddata(points=(self.data.coor_x, self.data.coor_y),
                                        values=self.data.sig_xy,
                                        xi=(self.x_grid, self.y_grid),
                                        method='linear')
        self.interp_disp_x = griddata(points=(self.data.coor_x, self.data.coor_y),
                                      values=self.data.disp_x - disp_x_0_0,
                                      xi=(self.x_grid, self.y_grid),
                                      method='linear')
        self.interp_disp_y = griddata(points=(self.data.coor_x, self.data.coor_y),
                                      values=self.data.disp_y - disp_y_0_0,
                                      xi=(self.x_grid, self.y_grid),
                                      method='linear')

    def optimize_cjp_displacements(self, method='lm', init_coeffs=None):
        """Optimizes CJP displacements.
        (see Yang et al. (2021) New algorithm for optimised fitting of DIC data
        to crack tip plastic zone using the CJP model)

        Args:
            method: method from scipy.optimize.least_squares, defaults to 'lm' - Levenberg-Marquardt iterative algorithm
            init_coeffs: initial coefficients used for x0 in scipy.optimize.least_squares

        """
        if init_coeffs is None:
            init_coeffs = np.random.rand(5)
        else:
            init_coeffs += np.random.rand(5)
        # optimize least squares
        return optimize.least_squares(fun=self.residuals_cjp_displacements,
                                      x0=init_coeffs,
                                      method=method)

    def optimize_williams_displacements(self, method='lm', init_coeffs=None):
        """Optimizes Williams displacements.
        (see Yang et al. (2021) New algorithm for optimised fitting of DIC data
        to crack tip plastic zone using the CJP model)

        Args:
            method: method from scipy.optimize.least_squares, defaults to 'lm' - Levenberg-Marquardt iterative algorithm
            init_coeffs: initial coefficients used for x0 in scipy.optimize.least_squares

        """
        if init_coeffs is None:
            init_coeffs = np.random.rand(2*len(self.terms))

        # optimize least squares
        return optimize.least_squares(fun=self.residuals_williams_displacements,
                                      x0=init_coeffs,
                                      method=method)

    def optimize_williams_stresses(self, method='lm', init_coeffs=None):
        """Optimizes Williams stresses.

        Args:
            method: method from scipy.optimize.least_squares, defaults to 'lm' - Levenberg-Marquardt iterative algorithm
            init_coeffs: initial coefficients used for x0 in scipy.optimize.least_squares

        """
        if init_coeffs is None:
            init_coeffs = np.random.rand(2*len(self.terms))
        else:
            init_coeffs += np.random.rand(2*len(self.terms))

        return optimize.least_squares(fun=self.residuals_williams_stresses,
                                      x0=init_coeffs,
                                      method=method)

    def residuals_cjp_displacements(self, inp: list or np.array) -> np.ndarray:
        """Returns the residuals of CJP displacements.

        Args:
            inp: coefficients for cjp_displacement_field, Z = (A_r, B_r, B_i, C, E) as in Christopher et al. '13

        Returns:
            residual: of cjp displacements, i.e. [cjp_displacement_x - measured_displacement_x,
                                                  cjp_displacement_y - measured_displacement_y]

        """
        z = inp

        cjp_disp_x, cjp_disp_y = cjp_displ_field(z, self.phi_grid, self.r_grid, self.material)

        residual = np.asarray([cjp_disp_x - self.interp_disp_x, cjp_disp_y - self.interp_disp_y])
        residual = residual.reshape(-1)
        return residual

    def residuals_williams_displacements(self, inp: list or np.array) -> np.ndarray:
        """Returns the residuals of Williams displacements.

        Args:
            inp: Williams coefficients for williams_displ_field

        Returns:
            residual: of displacements calculated from the approximated Williams field and the actual results

        """
        a = inp[0:len(self.terms)]
        b = inp[len(self.terms):]

        williams_disp_x, williams_disp_y = williams_displ_field(a, b, self.terms, self.phi_grid, self.r_grid, self.material)

        residual = np.asarray([williams_disp_x - self.interp_disp_x, williams_disp_y - self.interp_disp_y])
        residual = residual.reshape(-1)
        return residual

    def residuals_williams_stresses(self, inp: list or np.array) -> np.ndarray:
        """Returns the residuals of Williams stresses.

        Args:
            inp: Williams coefficients for williams_stress_field

        Returns:
            residual: between stresses calculated from approximated Williams field and actual results.
                      Calculated for x and y direction and xy shear stress.

        """
        a = inp[:len(self.terms)]
        b = inp[len(self.terms):]

        williams_sig_x, williams_sig_y, williams_sig_xy = williams_stress_field(a, b, self.terms,
                                                                                self.phi_grid, self.r_grid)

        residual = np.asarray([williams_sig_x - self.interp_sigma_x,
                               williams_sig_y - self.interp_sigma_y,
                               williams_sig_xy - self.interp_sigma_xy])
        residual = residual.reshape(-1)
        return residual

    def sse_williams_displacements(self, inp: list or np.array) -> float:
        """Returns the summed squared error of Williams displacements (sum over all entries and summed over
        x and y directions).

        Args:
            inp: Williams coefficients for williams_displ_field

        Returns:
            err

        """
        a = inp[:len(self.terms)]
        b = inp[len(self.terms):]

        williams_disp_x, williams_disp_y = williams_displ_field(a, b, self.terms, self.phi_grid, self.r_grid,
                                                                self.material)

        err = np.sum((williams_disp_x - self.interp_disp_x)**2 +
                     (williams_disp_y - self.interp_disp_y)**2)
        return err

    def sse_williams_stresses(self, inp: list or np.array) -> float:
        """Returns the summed squared error of Williams stresses (sum over x, y and xy directions).

        Args:
            inp: Williams coefficients for williams_stress_field

        Returns:
            err:

        """
        a = inp[:len(self.terms)]
        b = inp[len(self.terms):]

        williams_sig_x, williams_sig_y, williams_sig_xy = williams_stress_field(a, b, self.terms,
                                                                                self.phi_grid, self.r_grid)

        err = np.sum((williams_sig_x - self.interp_sigma_x)**2 +
                     (williams_sig_y - self.interp_sigma_y)**2 +
                     (williams_sig_xy - self.interp_sigma_xy)**2)
        return err

    @staticmethod
    def make_cartesian(r: float, phi: float):
        """Takes polar coordinates and maps onto cartesian coordinates."""
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return x, y
