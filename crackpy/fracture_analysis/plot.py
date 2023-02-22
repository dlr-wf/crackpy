import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.crack_tip import cjp_displ_field, williams_displ_field


class PlotSettings:
    def __init__(self, xlim_down: float = None, xlim_up: float = None, ylim_down: float = None, ylim_up: float = None,
                 background: str = 'eps_vm', min_value: float = None, max_value: float = None, extend: str = None):
        """Define plot settings for Plotter class object.

        Args:
            xlim_down: lower limit of x-axis
            xlim_up: upper limit of x-axis
            ylim_down: bottom of plot in [mm] (negative)
            ylim_up: top of plot in [mm]
            background: background plotted (e.g. 'sig_vm', 'eps_vm', 'disp_y', 'disp_x')
            min_value: minimum value of background
            max_value: maximum value of background
            extend: extend of background (e.g. 'neither', 'min', 'max')

        """
        self.ylim_down = ylim_down
        self.ylim_up = ylim_up
        self.xlim_down = xlim_down
        self.xlim_up = xlim_up

        self.background = background
        self.min_value = min_value
        self.max_value = max_value
        self.extend = extend
        self.legend_label = self._keyword_to_legend_label(background)

    def _keyword_to_legend_label(self, keyword: str = 'sig_vm') -> str:
        """Convert keyword to legend label.

        Args:
            keyword: keyword to convert to legend label

        Returns:
            legend label

        """
        if keyword == 'sig_vm':
            return 'Von Mises stress $\\sigma_{vm}$'
        if keyword == 'eps_vm':
            return 'Von Mises strain $\\varepsilon_{vm}$'
        if keyword == 'disp_x':
            return 'x-displacement $u_x$'
        if keyword == 'disp_y':
            return 'y-displacement $u_y$'
        if keyword == 'eps_x':
            return 'Strain $\\varepsilon_{xx}$'
        if keyword == 'eps_y':
            return 'Strain $\\varepsilon_{yy}$'
        if keyword == 'eps_xy':
            return 'Strain $\\varepsilon_{xy}$'
        if keyword == 'sig_x':
            return 'Stress $\\sigma_{xx}$'
        if keyword == 'sig_y':
            return 'Stress $\\sigma_{yy}$'
        if keyword == 'sig_xy':
            return 'Stress $\\sigma_{xy}$'

        print(f"Warning: keyword {keyword} not recognized. Using 'sig_vm' instead.")
        self.background = 'sig_vm'
        return 'Von Mises stress $\\sigma_{vm}$'


class Plotter:
    """Make plots of Fracture Analysis results and integration paths.

    Methods
        * plot - create and save the plot

    """

    def __init__(
            self,
            path: str,
            fracture_analysis: FractureAnalysis,
            plot_sets: PlotSettings
    ):
        """Initialize Plotter arguments.

        Args:
            path: path to output file
            fracture_analysis: fracture analysis results to be plotted
            plot_sets: plot settings used

        """
        self.analysis = fracture_analysis
        self.plot_sets = plot_sets

        self.path = self._make_path(path)
        self.filename = self._set_filename()

        self.figure, self.ax_results, self.ax_williams_opt, self.ax_cjp_opt, self.ax_int = self._plot_base_figure()

    def plot(self):
        """Main function to plot and save Fracture Analysis results."""
        # Plot results as text boxes
        self._plot_results()

        if self.analysis.optimization_properties is not None \
                and self.analysis.res_cjp is not None \
                and self.analysis.sifs_fit is not None:

            # Plot Williams fitting error
            self._plot_williams_residuals()

            # Plot CJP fitting error
            self._plot_cjp_residuals()

        if self.analysis.integral_properties is not None:
            # Plot integration
            self._plot_integration()

        # Save figure
        save_path = os.path.join(self.path, self.filename)
        plt.savefig(save_path + '.png', bbox_inches='tight')
        plt.clf()
        plt.close()

    @staticmethod
    def _plot_base_figure():
        """Plots the background around the centered crack tip without integration paths."""
        # Diagram
        plt.clf()

        gs_kw = dict(width_ratios=[1, 1.5, 2], height_ratios=[1, 1])
        fig, axs = plt.subplot_mosaic([['left', 'upper_middle', 'right'],
                                       ['left', 'lower_middle', 'right']],
                                      figsize=(16, 9), gridspec_kw=gs_kw)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

        # name axes
        ax_results = axs['left']
        ax_williams_opt = axs['upper_middle']
        ax_cjp_opt = axs['lower_middle']
        ax_int = axs['right']

        # turn off axes by default
        ax_results.axis('off')
        ax_williams_opt.axis('off')
        ax_cjp_opt.axis('off')
        ax_int.axis('off')

        return fig, ax_results, ax_williams_opt, ax_cjp_opt, ax_int

    def _plot_integration(self):
        self.ax_int.set_axis_on()

        # Background and min / max values
        background = getattr(self.analysis.data, self.plot_sets.background)
        min_value = background.min() if self.plot_sets.min_value is None else self.plot_sets.min_value
        max_value = background.max() if self.plot_sets.max_value is None else self.plot_sets.max_value
        extend = 'neither' if self.plot_sets.extend is None else self.plot_sets.extend

        contour_vector = np.linspace(min_value, max_value, 120, endpoint=True)
        labels = np.linspace(min_value, max_value, 5, endpoint=True)

        self.ax_int.set_title('Integral evaluation nodes', fontsize=16)

        # Set general font size
        plt.rcParams['font.size'] = '16'

        plot = self.ax_int.tricontourf(self.analysis.data.coor_x, self.analysis.data.coor_y, background,
                                       contour_vector, extend=extend)
        ct_plot = self.ax_int.scatter(0, 0, marker='x', color='black', s=100, linewidth=3)
        self.ax_int.legend(handles=[ct_plot],
                           labels=[f'$x_{{tip}}$ = {self.analysis.crack_tip.crack_tip_x:.2f} mm\n'
                                   f'$y_{{tip}}$ = {self.analysis.crack_tip.crack_tip_y:.2f} mm\n'
                                   f'$\\alpha_{{tip}}$ = {self.analysis.crack_tip.crack_tip_angle:.2f}°'],
                           loc='upper right', prop={'size': 12})

        divider = make_axes_locatable(self.ax_int)
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        plt.colorbar(plot, ticks=labels, cax=cax, orientation='horizontal', label=self.plot_sets.legend_label)

        # Set tick font size
        for label in (self.ax_int.get_xticklabels() + self.ax_int.get_yticklabels()):
            label.set_fontsize(16)

        # Set axis
        self.ax_int.axis('image')

        # Set axis limits
        left = self.analysis.integral_properties.integral_size_left
        right = self.analysis.integral_properties.integral_size_right
        bottom = self.analysis.integral_properties.integral_size_bottom
        top = self.analysis.integral_properties.integral_size_top
        step_left = self.analysis.integral_properties.paths_distance_left
        step_right = self.analysis.integral_properties.paths_distance_right
        step_bottom = self.analysis.integral_properties.paths_distance_bottom
        step_top = self.analysis.integral_properties.paths_distance_top
        num_paths = self.analysis.integral_properties.number_of_paths

        if self.plot_sets.xlim_down is not None:
            x_down = self.plot_sets.xlim_down
        else:
            x_down = left - step_left * (num_paths + 1)
        if self.plot_sets.xlim_up is not None:
            x_up = self.plot_sets.xlim_up
        else:
            x_up = right + step_right * (num_paths + 1)
        if self.plot_sets.ylim_down is not None:
            y_down = self.plot_sets.ylim_down
        else:
            y_down = bottom - step_bottom * (num_paths + 1)
        if self.plot_sets.ylim_up is not None:
            y_up = self.plot_sets.ylim_up
        else:
            y_up = top + step_top * (num_paths + 1)

        self.ax_int.set_xlim([x_down, x_up])
        self.ax_int.set_ylim([y_down, y_up])

        # Plot integration points
        for int_points in self.analysis.integration_points:
            self.ax_int.scatter(int_points[0], int_points[1], linewidths=0.1, marker='.', color='gray')

    def _plot_results(self):
        # main results as text boxes
        self.ax_results.set_axis_off()

        if self.analysis.integral_properties is not None:

            # J integral result
            props = dict(boxstyle='round', facecolor='gray', alpha=0.4)
            text = "J-integral\n\n" + \
                   f"$J$ = {self.analysis.sifs_int['rej_out_mean']['j']:.2f} $N*mm^{{-1}}$\n" + \
                   f"$K_J$ = {self.analysis.sifs_int['rej_out_mean']['sif_j']:.2f} $MPa*m^{{1/2}}$"
            self.ax_results.text(0.1, 0.97, text.replace('*', '\\cdot '),
                                 transform=self.ax_results.transAxes, fontsize=14,
                                 verticalalignment='top', bbox=props)

            # Williams integration results
            props = dict(boxstyle='round', facecolor='gray', alpha=0.4)
            text = "Interaction integral\n\n" + \
                   f"$K_I$ = {self.analysis.sifs_int['rej_out_mean']['sif_k_i']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$K_{{II}}$ = {self.analysis.sifs_int['rej_out_mean']['sif_k_ii']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$T$ = {self.analysis.sifs_int['rej_out_mean']['t_stress_int']:.2f} $MPa$"
            self.ax_results.text(0.1, 0.8, text.replace('*', '\\cdot '),
                                 transform=self.ax_results.transAxes, fontsize=14,
                                 verticalalignment='top', bbox=props)

            props = dict(boxstyle='round', facecolor='gray', alpha=0.4)
            text = "Bueckner integral\n\n" + \
                   f"$K_I$ = {self.analysis.sifs_int['rej_out_mean']['k_i_chen']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$K_{{II}}$ = {self.analysis.sifs_int['rej_out_mean']['k_ii_chen']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$T$ = {self.analysis.sifs_int['rej_out_mean']['t_stress_chen']:.2f} $MPa$"
            self.ax_results.text(0.1, 0.6, text.replace('*', '\\cdot '),
                                 transform=self.ax_results.transAxes, fontsize=14,
                                 verticalalignment='top', bbox=props)

        if self.analysis.optimization_properties is not None \
                and self.analysis.sifs_fit is not None \
                and self.analysis.res_cjp is not None:

            # Williams fitting results
            props = dict(boxstyle='round', facecolor='gray', alpha=0.4)
            text = "Williams fitting\n\n" + \
                   f"$K_I$ = {self.analysis.sifs_fit['K_I']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$K_{{II}}$ = {self.analysis.sifs_fit['K_II']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$T$ = {self.analysis.sifs_fit['T']:.2f} $MPa$"
            self.ax_results.text(0.1, 0.4, text.replace('*', '\\cdot '),
                                 transform=self.ax_results.transAxes, fontsize=14,
                                 verticalalignment='top', bbox=props)

            # CJP fitting results
            props = dict(boxstyle='round', facecolor='gray', alpha=0.4)
            text = "CJP fitting\n\n" + \
                   f"$K_F$ = {self.analysis.res_cjp['K_F']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$K_R$ = {self.analysis.res_cjp['K_R']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$K_S$ = {self.analysis.res_cjp['K_S']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$K_{{II}}$ = {self.analysis.res_cjp['K_II']:.2f} $MPa*m^{{1/2}}$\n" + \
                   f"$T$ = {self.analysis.res_cjp['T']:.2f} $MPa$"
            self.ax_results.text(0.1, 0.2, text.replace('*', '\\cdot '),
                                 transform=self.ax_results.transAxes, fontsize=14,
                                 verticalalignment='top', bbox=props)

    def _plot_cjp_residuals(self):
        self.ax_cjp_opt.set_axis_on()

        opt = self.analysis.optimization
        cjp_disp_x, cjp_disp_y = cjp_displ_field(self.analysis.cjp_coeffs, opt.phi_grid, opt.r_grid, opt.material)
        residuals = np.asarray([cjp_disp_x - opt.interp_disp_x, cjp_disp_y - opt.interp_disp_y])
        error = np.sqrt(np.sum(residuals**2, axis=0))

        # Set general font size
        plt.rcParams['font.size'] = '16'

        plot = self.ax_cjp_opt.scatter(opt.x_grid.flatten(), opt.y_grid.flatten(), c=error, marker='.')

        # Set tick font size
        for label in (self.ax_cjp_opt.get_xticklabels() + self.ax_cjp_opt.get_yticklabels()):
            label.set_fontsize(16)

        self.ax_cjp_opt.set_xlim([self.plot_sets.xlim_down, self.plot_sets.xlim_up])
        self.ax_cjp_opt.set_ylim([self.plot_sets.ylim_down, self.plot_sets.ylim_up])
        self.ax_cjp_opt.axis('image')

        divider = make_axes_locatable(self.ax_cjp_opt)
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
        labels = np.linspace(error.flatten().min(), error.flatten().max(), 2, endpoint=True)
        plt.colorbar(plot, ticks=labels, cax=cax, orientation='horizontal', label='CJP fitting error', format='%.0e')

    def _plot_williams_residuals(self):
        self.ax_williams_opt.set_axis_on()

        opt = self.analysis.optimization
        a = self.analysis.williams_coeffs[:len(opt.terms)]
        b = self.analysis.williams_coeffs[len(opt.terms):]
        will_disp_x, will_disp_y = williams_displ_field(a, b, opt.terms,
                                                        opt.phi_grid, opt.r_grid, opt.material)
        residuals = np.asarray([will_disp_x - opt.interp_disp_x, will_disp_y - opt.interp_disp_y])
        error = np.sqrt(np.sum(residuals ** 2, axis=0))

        # Set general font size
        plt.rcParams['font.size'] = '16'

        plot = self.ax_williams_opt.scatter(opt.x_grid.flatten(), opt.y_grid.flatten(), c=error, marker='.')

        # Set tick font size
        for label in (self.ax_williams_opt.get_xticklabels() + self.ax_williams_opt.get_yticklabels()):
            label.set_fontsize(16)

        self.ax_williams_opt.set_xlim([self.plot_sets.xlim_down, self.plot_sets.xlim_up])
        self.ax_williams_opt.set_ylim([self.plot_sets.ylim_down, self.plot_sets.ylim_up])
        self.ax_williams_opt.axis('image')

        divider = make_axes_locatable(self.ax_williams_opt)
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
        labels = np.linspace(error.flatten().min(), error.flatten().max(), 2, endpoint=True)
        plt.colorbar(plot, ticks=labels, cax=cax, orientation='horizontal', label='Williams fitting error', format='%.0e')

    @staticmethod
    def _make_path(path):
        """Create and return path."""
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _set_filename(self) -> str:
        """Returns filename with '_side' at the end. E.g. 'filename.txt' -> 'filename_side'."""
        return os.path.split(self.analysis.nodemap_file)[-1][:-4] + '_' + self.analysis.crack_tip.left_or_right
