import logging
from pathlib import Path

import numpy as np
import scipy
from scipy import optimize
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from crackpy.input.input_data import InputData

logger = logging.getLogger(__name__)


class CrackDetectionLineIntercept:
    """Crack detection class using a line interception method.

    This method fits a tanh function to displacement data on vertical slices
    to detect the crack path and tip position.

    Attributes:
        data: Input nodemap data
        x_min: Minimum x-coordinate of detection window
        x_max: Maximum x-coordinate of detection window
        y_min: Minimum y-coordinate of detection window
        y_max: Maximum y-coordinate of detection window
        tick_size_x: Tick size in x direction
        tick_size_y: Tick size in y direction
        grid_component: Displacement component for detection ('ux' or 'uy')
        eps_vm_threshold: Threshold for crack detection
        window_size: Size of sliding window for thresholding
        angle_estimation_mm_radius: Radius for crack angle estimation in mm
        crack_tip: Detected crack tip position
        crack_path: Detected crack path
        crack_angle: Detected crack angle

    Methods:
        * predict_tip_pos - predict the crack tip position using line interception method

    """

    def __init__(
            self,
            data: InputData,
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            tick_size_x: float = 0.1,
            tick_size_y: float = 0.1,
            grid_component: str = 'uy',
            eps_vm_threshold: float = 0.01,
            window_size: float = 3,
            angle_estimation_mm_radius: float = 50
    ) -> None:
        """Initialize class arguments.

        Args:
            data: input data
            x_min: minimum x coordinate of the crack detection window
            x_max: maximum x coordinate of the crack detection window
            y_min: minimum y coordinate of the crack detection window
            y_max: maximum y coordinate of the crack detection window
            tick_size_x: tick size in x direction
            tick_size_y: tick size in y direction
            grid_component: displacement component to use for crack detection (ux or uy)
            eps_vm_threshold: threshold for crack detection used along the detected crack path
            window_size: size of the sliding window used for thresholding
            angle_estimation_mm_radius: mm radius used for crack angle estimation

        """
        self.tip_index = 0
        self.eps_vm_crack_path = None
        self.x_path = None
        self.y_path = None
        self.coefficients_fitted = None
        self.data = data
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.tick_size_x = tick_size_x
        self.tick_size_y = tick_size_y
        self.grid_component = grid_component
        self.window_size = window_size
        self.eps_vm_threshold = eps_vm_threshold
        self.angle_estimation_mm_radius = angle_estimation_mm_radius

        # crack detection results
        self.crack_tip = None
        self.crack_path = None
        self.crack_angle = None

        # crack tip correction
        self.ct_corr_rethore = None
        self.ct_corr_grid = None
        self.ct_corr_opt = None
        self.df_grid_errors = None

        # map displacements to separate grid
        self._map_data_to_grid()

        logger.debug("CrackDetectionLineIntercept initialized: window=[%s,%s]x[%s,%s], grid_component=%s, eps_vm_threshold=%s",
                     x_min, x_max, y_min, y_max, grid_component, eps_vm_threshold)

    def run(self) -> None:
        """Run crack detection with line intercept method."""
        logger.debug("Starting line intercept crack detection with eps_vm_threshold=%.4f, window_size=%s", self.eps_vm_threshold, self.window_size)

        # Fit a Formula to each slice of the grid
        coefficients_fitted = []

        # The init_coeff is the y-coordinate of the crack path at the x-coordinate of the current slice.
        # A good initial guess is helpful for the optimiser.
        init_coeff = self.y_min + (self.y_max - self.y_min) / 2.0

        self.x_path = []
        num_valid_slices = 0
        for step, x_coordinate in enumerate(self.x_coords):
            if not np.isnan(self.disp_grid[:, step]).all():
                num_valid_slices += 1
                init_coeffs = [1.0, init_coeff, 1.0, 0.0, 0.0]
                res = optimize.least_squares(
                    fun=self._residuals_tanh,
                    x0=init_coeffs,
                    args=(self.y_coords, self.disp_grid[:, step]),
                    bounds=([-np.inf, self.y_min, -np.inf, -np.inf, -np.inf], [np.inf, self.y_max, np.inf, np.inf, np.inf]),
                    method='trf'
                )
                fitted_coefficients = res.x

                # If the fitted y-coordinate is within the y-range, update the init_coeff
                if fitted_coefficients[1] > self.y_min and fitted_coefficients[1] < self.y_max:
                    init_coeff = fitted_coefficients[1]

                coefficients_fitted.append(fitted_coefficients)
                self.x_path.append(x_coordinate)

        logger.debug("Fitted tanh function to %d valid slices out of %d total slices", num_valid_slices, len(self.x_coords))

        self.coefficients_fitted = np.asarray(coefficients_fitted).T

        # The y-coordinate of the crack path corresponds to the coefficient "B".
        self.y_path = np.asarray(self.coefficients_fitted[1, :])


        # Find the crack tip
        self.eps_vm_crack_path = scipy.interpolate.griddata((self.data.coor_x, self.data.coor_y), self.data.eps_vm,
                                                       (self.x_path, self.y_path), method='linear')
        self.tip_index = 0
        reversed_eps_vm_crack_path = self.eps_vm_crack_path[::-1]
        for i in range(len(reversed_eps_vm_crack_path) - self.window_size + 1):
            if np.all(reversed_eps_vm_crack_path[i:i + self.window_size] > self.eps_vm_threshold):
                self.tip_index = len(reversed_eps_vm_crack_path) - i - 1
                break

        logger.debug("Crack tip search: tip_index=%d out of %d path points", self.tip_index, len(self.x_path))

        if self.tip_index > 0:
            self.crack_tip = np.asarray([self.x_path[self.tip_index], self.y_path[self.tip_index]])
            self.crack_path = np.stack([self.x_path[0:self.tip_index], self.y_path[0:self.tip_index]], axis=-1)

            # crack angle estimation
            angle_estimation_px_radius = int(self.angle_estimation_mm_radius / self.tick_size_x)

            # linear fit near crack tip
            x = self.crack_path[-angle_estimation_px_radius:-1, 0]
            y = self.crack_path[-angle_estimation_px_radius:-1, 1]
            line_coeffs = np.polyfit(x, y, 1)
            m = line_coeffs[0]
            c = line_coeffs[1]
            yy = m * x + c
            self.crack_angle = np.arctan2(yy[-1] - yy[0], x[-1] - x[0]) * 180.0 / np.pi

            logger.debug("Crack tip detected at (%.3f, %.3f) mm, angle=%.2f°", self.crack_tip[0], self.crack_tip[1], self.crack_angle)
            logger.debug("Crack path contains %d points", len(self.crack_path))
        else:
            self.crack_tip = np.asarray([np.nan, np.nan])
            self.crack_path = np.stack([np.nan, np.nan], axis=-1)
            self.crack_angle = np.nan
            logger.warning('No crack tip detected by line intercept method; returning NaN values.')

    def plot(self, fname: str, folder: str, crack_tip_results: dict= None,
             crack_tip_position: dict = None, fmin: float = 0, fmax: float = 0.0068, plot_window: list = None):
        """Plot the von Mises strain and the corresponding crack detection results and save under `fname` in `folder`.

        Args:
            fname: filename
            folder: folder
            crack_tip_results: crack tip correction results (e.g. from Rethore method, optimization, or grid search)
                            Example: {'Rethore': [dx, dy], 'Grid Search': [dx, dy], 'Optimization': [dx, dy]}
            crack_tip_position: crack tip position (e.g. absolute crack tip position)
            fmin: minimum value for the colorbar
            fmax: maximum value for the colorbar
            plot_window: list with [x_min, x_max, y_min, y_max] to plot a window.
                         If None: self.x_min, self.x_max, self.y_min, self.y_max is used

        """
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)

        num_colors = 120
        contour_vector = np.linspace(fmin, fmax, num_colors, endpoint=True)
        label_vector = np.linspace(fmin, fmax, 10, endpoint=True)

        plt.clf()
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        plot = ax.tricontourf(self.data.coor_x, self.data.coor_y, self.data.sig_vm, contour_vector, extend='max')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(plot, ticks=label_vector, cax=cax, label='Von Mises stress')
        if self.crack_path is not None:
            ax.scatter(self.crack_path[:, 0], self.crack_path[:, 1], color='black', s=1, marker='.')
        ax.scatter(self.crack_tip[0], self.crack_tip[1], color='black', linewidths=1, marker='x', label='Crack tip')

        if crack_tip_results is not None:
            for method, ct_corr in crack_tip_results.items():
                ax.scatter(self.crack_tip[0] + ct_corr[0], self.crack_tip[1] + ct_corr[1],
                           linewidths=1, marker='x', label=method)

        if crack_tip_position is not None:
            for method, ct_pos in crack_tip_position.items():
                ax.scatter(ct_pos[0], ct_pos[1],
                           linewidths=1, marker='x', label=method)

        ax.legend(loc='upper left')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.axis('image')

        if plot_window is None:
            ax.set_xlim([self.x_min, self.x_max])
            ax.set_ylim([self.y_min, self.y_max])
        else:
            ax.set_xlim([plot_window[0], plot_window[1]])
            ax.set_ylim([plot_window[2], plot_window[3]])

        plt.savefig(str(p / fname), bbox_inches='tight')

    def _map_data_to_grid(self):
        """Map the data to a grid."""
        steps_x = int((self.x_max - self.x_min) / self.tick_size_x)
        steps_y = int((self.y_max - self.y_min) / self.tick_size_y)
        logger.debug("Mapping data to grid: %dx%d grid points", steps_x, steps_y)
        logger.debug("Grid range: x=[%.2f, %.2f], y=[%.2f, %.2f]", self.x_min, self.x_max, self.y_min, self.y_max)

        self.x_coords = np.linspace(self.x_min, self.x_max, steps_x, endpoint=True)
        self.y_coords = np.linspace(self.y_min, self.y_max, steps_y, endpoint=True)
        self.x_grid, self.y_grid = np.meshgrid(self.x_coords, self.y_coords)
        self.disp_x_grid = scipy.interpolate.griddata((self.data.coor_x, self.data.coor_y), self.data.disp_x,
                                                      (self.x_grid, self.y_grid), method='linear')
        self.disp_y_grid = scipy.interpolate.griddata((self.data.coor_x, self.data.coor_y), self.data.disp_y,
                                                      (self.x_grid, self.y_grid), method='linear')
        if self.grid_component == 'ux':
            self.disp_grid = self.disp_x_grid
        if self.grid_component == 'uy':
            self.disp_grid = self.disp_y_grid

        logger.debug("Grid component '%s' mapped, range: [%.4f, %.4f] mm", self.grid_component, np.nanmin(self.disp_grid), np.nanmax(self.disp_grid))

    def _tanh_funct(self, coefficients, coordinates):
        """Hyperbolic tangent function to approximate the displacements.
           The y-coordinate of the crack path corresponds to the coefficient "B".

        Args:
            coefficients: coefficients
            coordinates: coordinates
        Returns:
            discplacements_fit: fitted displacements

        """
        A, B, C, D, E = coefficients
        discplacements_fit = A * np.tanh((coordinates - B) * C) + D + E * coordinates
        return discplacements_fit

    def _residuals_tanh(self, coefficients, coordinates, displacements_dic):
        """Residuals for the tanh function.

        Args:
            coefficients: coefficients
            coordinates: coordinates
            displacements_dic: displacements
        Returns:
            res: residuals

        """
        res = displacements_dic - self._tanh_funct(coefficients, coordinates)
        return res


def plot_grid_errors(df, fname: str, folder: str):
    """Plot the grid search results and save it under `fname` in `folder`.

    Args:
        df: dataframe with grid search results (Expected: first column: dx, second column: dy, third column: error)
        fname: filename
        folder: folder

    """
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)

    dx = df.iloc[:, 0]
    dy = df.iloc[:, 1]
    error = df.iloc[:, 3]

    # min error index
    min_error_index = error.idxmin()

    num_colors = 120
    min_contour = 0.0
    max_contour = error.max()

    contour_vector = np.linspace(min_contour, max_contour, num_colors, endpoint=True)
    label_vector = np.linspace(min_contour, max_contour, 10, endpoint=True)

    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plot = ax.tricontourf(dx, dy, error, contour_vector)
    ax.scatter(dx[min_error_index], dy[min_error_index], color='black', marker='x', s=100)
    ax.scatter(dx, dy, color='black', marker='o', s=1)

    plt.colorbar(plot, ticks=label_vector, label='error')

    ax.set_xlabel('dx [mm]')
    ax.set_ylabel('dy [mm]')
    ax.axis('image')
    # Save figure
    plt.savefig(str(p / fname), bbox_inches='tight')
