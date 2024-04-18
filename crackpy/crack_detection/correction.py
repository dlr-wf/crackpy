import copy
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import optimize
from crackpy.fracture_analysis.optimization import Optimization, OptimizationProperties


def run_williams_optimization(data, material, opt_props):
    """Run Williams optimization for a given crack tip position.

    Args:
        data: InputData
        material: material properties
        opt_props: optimization properties
    Returns:
        williams_fit_a_n: Williams coefficients a_n
        williams_fit_b_n: Williams coefficients b_n
        error: error of the optimization is res.cost * 1000

    """
    # run Williams optimization
    optimization = Optimization(data=data, options=opt_props, material=material)

    # calculate Williams coefficients with fitting method
    res = optimization.optimize_williams_displacements()
    williams_coeffs = res.x
    a_n = williams_coeffs[:len(optimization.terms)]
    b_n = williams_coeffs[len(optimization.terms):]
    williams_fit_a_n = {n: a_n[index] for index, n in enumerate(optimization.terms)}
    williams_fit_b_n = {n: b_n[index] for index, n in enumerate(optimization.terms)}

    error = res.cost

    return williams_fit_a_n, williams_fit_b_n, error


class CrackTipCorrection:
    """Class for the correction of crack tip position."""

    def __init__(self, data, crack_tip, crack_angle, material):
        self.data = data
        self.crack_tip = crack_tip
        self.crack_angle = crack_angle
        self.material = material
        self.iteration_log = pd.DataFrame()

    def correct_crack_tip(
            self,
            opt_props,
            max_iter=100,
            step_tol=1e-3,
            damper=1,
            method='rethore',
            verbose=False,
            plot_intermediate_results=False,
            cd=None,
            folder=None,
            d_x_str=None,
            d_y_str=None,
    ):
        """Find exact crack tip position with the method described in
        Rethore, J. (2015). "Automatic crack tip detection and stress intensity factors estimation..."

        The method is based on fitting the Williams expansion to the data and correcting the crack tip position on the
        support of the crack path. Therefore, the crack angle and initial crack tip position are required. The tip
        position is iteratively corrected with the step

        .. math::
            \\Delta x = -2a_{-1}/a_1

        thus pushing the value of :math:`a_1` to zero. Choose method='rethore' to use this method.

        Alternatively, we learned a correction function from data based on the Williams coefficients up to order 1
        using symbolic regression. The correction function is then used to correct the crack tip position.
        Choose method='symbolic_regression' to use this method.

        The crack angle is not changed in both cases.

        Args:
            plot_intermediate_results:
            opt_props: OptimizationProperties
            max_iter: maximum number of iterations
            step_tol: tolerance for the step :math:`\\Delta x`
            damper: damper for the step size
            method: 'rethore', 'symbolic_regression' or 'custom_function'
            verbose: If True, print the current iteration
            d_x_str: If method='custom_function', provide a function for the correction in x as a string
            d_y_str: If method='custom_function', provide a function for the correction in y as a string
            folder: Plot folder
            cd: CrackDetectionIntercept object

        Returns:
            corrected crack tip position as list of x and y coordinates

        """
        # Initialize
        crack_tip_x = self.crack_tip[0]
        crack_tip_y = self.crack_tip[1]

        # iterate x-direction until convergence
        for i in range(max_iter):
            data_copy = copy.deepcopy(self.data)
            data_copy.transform_data(crack_tip_x, crack_tip_y, self.crack_angle)
            # Fit Williams expansion
            try:
                williams_fit_a_n, williams_fit_b_n, cost = run_williams_optimization(
                    data_copy, self.material, opt_props)
            except:
                print('Williams fit failed. No correction applied.')
                ct_corr = [0, 0]
                return ct_corr

            if method == 'rethore':
                # Rethore method
                d_x = -2 * williams_fit_a_n[-1] / williams_fit_a_n[1]
                d_y = 0
            elif method == 'symbolic_regression':
                # Symbolic regression method (learned correction function for mode I & mixed mode)
                # does not work for pure mode II
                d_x = - williams_fit_a_n[-1] / williams_fit_a_n[1]
                d_y = - williams_fit_b_n[-1] / williams_fit_a_n[1]
            elif method == 'custom_function':
                if d_x_str is None or d_y_str is None:
                    raise ValueError('Please provide a function for the correction in x and y as d_x_str and d_y_str.')
                # custom function from input string
                d_x = eval(d_x_str)
                d_y = eval(d_y_str)
            else:
                raise ValueError(f"Method {method} not supported. Choose 'rethore', 'symbolic_regression', "
                                 f"or 'custom_function'.")

            # damper
            d_x *= damper
            d_y *= damper

            # rotate shift vector
            d_x_rot, d_y_rot = self._rotate_data(d_x, d_y)

            # update crack tip position
            crack_tip_x += d_x_rot
            crack_tip_y += d_y_rot

            # plot intermediate results
            if plot_intermediate_results:
                assert folder is not None, 'Please provide a folder to save the plots.'
                assert cd is not None, 'Please provide a CrackDetectionIntercept object.'
                res = {
                    f"$a_{{-1}} = {williams_fit_a_n[-1]:.2f}, b_{{-1}} = {williams_fit_b_n[-1]:.2f}$":
                        [crack_tip_x - self.crack_tip[0], crack_tip_y - self.crack_tip[1]]
                }
                cd.plot(fname=f'iteration_{i}.png', folder=folder, crack_tip_results=res, fmax=self.material.sig_yield)

            if verbose:
                print(f"Iteration {i}: dx = {d_x_rot:+.4f}, dy = {d_y_rot:+.4f}, "
                      f"a_-1 = {williams_fit_a_n[-1]:.4f}, b_-1 = {williams_fit_b_n[-1]:.4f}, "
                      f"a_1 = {williams_fit_a_n[1]:.4f}, b_1 = {williams_fit_b_n[1]:.4f}, "
                      f"crack_tip_corrected = ({crack_tip_x:.4f}, {crack_tip_y:.4f})")

            # log iteration
            williams_dict = {}
            for key in williams_fit_a_n.keys():
                williams_dict[f'a_{key}'] = williams_fit_a_n[key]
            for key in williams_fit_b_n.keys():
                williams_dict[f'b_{key}'] = williams_fit_b_n[key]
            williams_df = pd.DataFrame(williams_dict, index=[i])

            process_log = pd.DataFrame({'Iteration': i, 'dx': d_x_rot, 'dy': d_y_rot,
                                        'crack_tip_x': crack_tip_x,
                                        'crack_tip_y': crack_tip_y}, index=[i])
            step_log = pd.concat([process_log, williams_df], axis=1)
            self.iteration_log = pd.concat([self.iteration_log, step_log], axis=0)

            # stop as soon as correction is smaller than the tolerance
            if np.sqrt(d_x_rot ** 2 + d_y_rot ** 2) < step_tol:
                break

        ct_corr = [crack_tip_x - self.crack_tip[0], crack_tip_y - self.crack_tip[1]]
        print(ct_corr)
        print('------------------------------------')
        return ct_corr

    def correct_crack_tip_optimization(
            self,
            opt_props: OptimizationProperties,
            objective: str = 'error',
            tol: float = 0.01,
            verbose: bool = False
    ):
        """Correct crack tip position using optimization of the Williams fitting error. The optimization is performed
        using the Levenberg-Marquardt algorithm for unconstrained least squares.

        Args:
            opt_props: optimization properties
            objective: objective function used for optimization ('error' or 'a_b_minus_one')
            tol: tolerance for optimization
            verbose: verbose output

        """
        if objective == 'error':
            objective_fn = self._fitting_error
        elif objective == 'a_b_minus_one':
            objective_fn = self._a_minus_one_b_minus_one_squared_error
        else:
            raise ValueError('Objective function not implemented.')

        init_coeffs = np.asarray([0.0, 0.0])
        res = optimize.minimize(
            fun=objective_fn,
            x0=init_coeffs,
            args=(opt_props, verbose),
            tol=tol
        )

        ct_corr = [res.x[0], res.x[1], 0]
        print(ct_corr)
        print('------------------------------------')
        return ct_corr

    def correct_crack_tip_differential_evolution(
            self,
            opt_props: OptimizationProperties,
            x_min: float = -2,
            x_max: float = 2,
            y_min: float = -2,
            y_max: float = 2,
            tol: float = 0.01,
            workers: int = 8,
            maxiter: int = 3,
            popsize: int = 4,
            verbose: bool = False
    ):
        """Correct crack tip position using optimization of the Williams fitting error. The optimization is performed
        using the differential evolution algorithm. This method has the distinct advantage that it is parallelizable.

        Args:
            opt_props: optimization properties
            x_min: minimum x coordinate for optimization
            x_max: maximum x coordinate for optimization
            y_min: minimum y coordinate for optimization
            y_max: maximum y coordinate for optimization
            tol: tolerance for optimization
            workers: number of workers for optimization (-1 = all available CPU cores)
            maxiter: maximum number of iterations for optimization
            popsize: population size for optimization
            verbose: verbose output

        """
        res = optimize.differential_evolution(
            func=self._fitting_error,
            bounds=([x_min, x_max], [y_min, y_max]),
            args=([opt_props, verbose]),
            init='latinhypercube',
            maxiter=maxiter,
            tol=tol,
            popsize=popsize,
            workers=workers,
            updating='deferred',
            polish=False,
            disp=True
        )

        ct_corr = [res.x[0], res.x[1], 0]
        print(ct_corr)
        print('------------------------------------')
        return ct_corr

    def _rotate_data(self, dx, dy):
        """Rotate data with the crack tip angle.

        Args:
            dx: x coordinate
            dy: y coordinate

        Returns:
            dx_rot: rotated x coordinate
            dy_rot: rotated y coordinate

        """
        angle = self.crack_angle / 180 * np.pi
        dx_rot = dx * np.cos(angle) - dy * np.sin(angle)
        dy_rot = dx * np.sin(angle) + dy * np.cos(angle)
        return dx_rot, dy_rot

    def _fitting_error(self, disp, opt_props, verbose=False):
        """Residuals for the crack tip position.

        Args:
            disp: crack tip displacement and rotation [dx, dy, dphi]
            opt_props: optimization properties

        Returns:
            error: error of fitting displacements to Williams expansion

        """
        data_copy = copy.deepcopy(self.data)
        data_copy.transform_data(self.crack_tip[0] + disp[0], self.crack_tip[1] + disp[1], self.crack_angle)
        williams_fit_a_n, williams_fit_b_n, error = run_williams_optimization(data_copy, self.material, opt_props)

        # scale error to avoid numerical issues
        error = error * 1000

        if verbose:
            print(f"dx = {disp[0]:+10.4f}, dy = {disp[1]:+10.4f}, error = {error:10.6f}, "
                  f"a_(-1) = {williams_fit_a_n[-1]:+10.3f}, "
                  f"b_(-1) = {williams_fit_b_n[-1]:+10.3f}")

        return error

    def _a_minus_one_b_minus_one_squared_error(self, disp, opt_props, verbose=False):
        """Residuals for the crack tip position.

        Args:
            disp: crack tip displacement and rotation [dx, dy, dphi]
            opt_props: optimization properties

        Returns:
            error: error of fitting displacements to Williams expansion

        """
        data_copy = copy.deepcopy(self.data)
        data_copy.transform_data(self.crack_tip[0] + disp[0], self.crack_tip[1] + disp[1], self.crack_angle)
        williams_fit_a_n, williams_fit_b_n, _ = run_williams_optimization(data_copy, self.material, opt_props)

        # squared error of a_(-1) and b_(-1)
        error = williams_fit_a_n[-1] ** 2 + williams_fit_b_n[-1] ** 2

        if verbose:
            print(f"dx = {disp[0]:+10.4f}, dy = {disp[1]:+10.4f}, error = {error:10.4f}, "
                  f"a_(-1) = {williams_fit_a_n[-1]:+10.3f}, "
                  f"b_(-1) = {williams_fit_b_n[-1]:+10.3f}")

        return error


class CrackTipCorrectionGridSearch:
    """Crack tip correction using brute force grid search."""

    def __init__(self, data, crack_tip, crack_angle, material):
        """Initialize the class.

        Args:
            data: data object
            crack_tip: crack tip position
            crack_angle: crack angle
            material: material properties
        """
        self.data = data
        self.crack_tip = crack_tip
        self.crack_angle = crack_angle
        self.material = material

    def correct_crack_tip_grid_search(
            self,
            opt_props: OptimizationProperties,
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            x_step: float,
            y_step: float,
            workers: int = 1,
            verbose: bool = False
    ):
        """Correct crack tip position using grid search of the smallest Williams fitting error.
        Warning! Bruteforce method with long runtime. Parallelized version.

        Args:
            opt_props: OptimizationProperties used for the Williams fitting
            x_min: minimum x coordinate for the grid search relative to the current crack tip position
            x_max: maximum x coordinate for the grid search relative to the current crack tip position
            y_min: minimum y coordinate for the grid search relative to the current crack tip position
            y_max: maximum y coordinate for the grid search relative to the current crack tip position
            x_step: step size in x direction
            y_step: step size in y direction
            workers: number of parallel jobs
            verbose: If True, print the current iteration

        Returns:
            crack tip correction as array of x and y coordinate, dataframe of error values for each grid point

        """
        delta_x = np.linspace(x_min, x_max, int((x_max - x_min) / x_step) + 1, endpoint=True)
        delta_y = np.linspace(y_min, y_max, int((y_max - y_min) / y_step) + 1, endpoint=True)
        delta_phi = 0
        results = []
        shifts_x_y = itertools.product(delta_x, delta_y)
        print(f"Number of grid points: {len(delta_x) * len(delta_y)}")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for shift_x_y in shifts_x_y:
                results.append(
                    executor.submit(self._parallel_grid_search, shift_x_y, delta_phi, opt_props, verbose))

        columns = ['dx', 'dy', 'dphi', 'error']
        for term in opt_props.terms:
            columns.append(f'a_{term}')
        for term in opt_props.terms:
            columns.append(f'b_{term}')

        df = pd.DataFrame(columns=columns)
        max_error = 1e10
        ct_corr = [0, 0]
        for i, result in enumerate(results):
            output = result.result()
            error = output[3]
            if error < max_error:
                max_error = error
                ct_corr = [output[0], output[1]]
            output[0] += self.crack_tip[0]
            output[1] += self.crack_tip[1]
            output[2] += self.crack_angle
            df.loc[i] = output

        print(ct_corr)
        print('------------------------------------')
        return ct_corr, df

    def _parallel_grid_search(self, shift_x_y, delta_phi, opt_props, verbose=False):
        """Process a single point in the grid search.

        Args:
            shift_x_y: shift in x and y direction
            delta_phi: delta, rotation angle
            opt_props: OptimizationProperties used for the Williams fitting
            verbose: If True, print the current iteration

        Returns:
            point with error value

        """
        dx, dy = shift_x_y
        data_copy = copy.deepcopy(self.data)
        data_copy.transform_data(self.crack_tip[0] + dx, self.crack_tip[1] + dy, self.crack_angle + delta_phi)

        williams_fit_a_n, williams_fit_b_n, error = run_williams_optimization(data_copy, self.material, opt_props)

        if verbose:
            print(f"Iteration: dx = {dx:+.4f}, dy = {dy:+.4f}, dphi = {delta_phi:+.4f} deg, error = {error:.8f}, "
                  f"a_-1 = {williams_fit_a_n[-1]}, b_-1 = {williams_fit_b_n[-1]}")

        output = [dx, dy, delta_phi, error]
        for term in opt_props.terms:
            output.append(williams_fit_a_n[int(term)])
        for term in opt_props.terms:
            output.append(williams_fit_b_n[int(term)])
        output = np.array(output)
        return output


class CustomCorrection(CrackTipCorrection):
    def __init__(self, data, crack_tip, crack_angle, material):
        super().__init__(data, crack_tip, crack_angle, material)

    def custom_correct_crack_tip(
            self,
            opt_props,
            dx_lambdified,
            dy_lambdified,
            max_iter=100,
            step_tol=1e-3,
            damper=1,
            verbose=False,
            plot_intermediate_results=False,
            cd=None,
            folder=None,
    ):
        """Find exact crack tip position iteratively using the Williams coefficient together with the provided
        sympy formulas for the correction in x and y.

        Important remark: The formulas are based on the Williams coefficients $A_{-3}$ to
        $A_7$ and $B_{-3}$ to $B_7$. Therefore skipping some of these terms in the Optimization Properties might lead to
        wrong correction results in the case when Williams coefficients used in the formulas are missing.

        Args:
            plot_intermediate_results:
            opt_props: OptimizationProperties
            dx_lambdified: lambdified sympy formula for the correction in x
            dy_lambdified: lambdified sympy formula for the correction in y
            max_iter: maximum number of iterations
            step_tol: tolerance for the step :math:`\\Delta x`
            damper: damper for the step size
            verbose: If True, print the current iteration
            folder: Plot folder
            cd: CrackDetectionIntercept object

        Returns:
            corrected crack tip position as list of x and y coordinates

        """
        if not opt_props.terms == [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]:
            print('Warning: The formulas are based on the Williams coefficients A_-3 to A_7 and B_-3 to B_7. '
                  'Therefore skipping some of these terms in the Optimization Properties might lead to'
                  'wrong correction results in case coefficients used in the formulas are missing.')
            missing_terms = list({-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7} - set(opt_props.terms))
            print(f'Missing terms: {missing_terms}')
        else:
            missing_terms = []

        # Initialize
        crack_tip_x = self.crack_tip[0]
        crack_tip_y = self.crack_tip[1]

        # iterate x-direction until convergence
        for i in range(max_iter):
            data_copy = copy.deepcopy(self.data)
            data_copy.transform_data(crack_tip_x, crack_tip_y, self.crack_angle)
            # Fit Williams expansion
            try:
                williams_fit_a_n, williams_fit_b_n, cost = run_williams_optimization(
                    data_copy, self.material, opt_props)
            except:
                print('Williams fit failed. No correction applied.')
                ct_corr = [0, 0]
                return ct_corr

            for term in missing_terms:
                # fill with zeros
                williams_fit_a_n[term] = 0
                williams_fit_b_n[term] = 0

            # custom function from input string
            d_x = dx_lambdified(
                williams_fit_a_n[-3], williams_fit_a_n[-2], williams_fit_a_n[-1], williams_fit_a_n[0],
                williams_fit_a_n[1], williams_fit_a_n[2], williams_fit_a_n[3], williams_fit_a_n[4],
                williams_fit_a_n[5], williams_fit_a_n[6], williams_fit_a_n[7],
                williams_fit_b_n[-3], williams_fit_b_n[-2], williams_fit_b_n[-1], williams_fit_b_n[0],
                williams_fit_b_n[1], williams_fit_b_n[2], williams_fit_b_n[3], williams_fit_b_n[4],
                williams_fit_b_n[5], williams_fit_b_n[6], williams_fit_b_n[7]
            )
            d_y = dy_lambdified(
                williams_fit_a_n[-3], williams_fit_a_n[-2], williams_fit_a_n[-1], williams_fit_a_n[0],
                williams_fit_a_n[1], williams_fit_a_n[2], williams_fit_a_n[3], williams_fit_a_n[4],
                williams_fit_a_n[5], williams_fit_a_n[6], williams_fit_a_n[7],
                williams_fit_b_n[-3], williams_fit_b_n[-2], williams_fit_b_n[-1], williams_fit_b_n[0],
                williams_fit_b_n[1], williams_fit_b_n[2], williams_fit_b_n[3], williams_fit_b_n[4],
                williams_fit_b_n[5], williams_fit_b_n[6], williams_fit_b_n[7]
            )

            # damper
            d_x *= damper
            d_y *= damper

            # rotate shift vector
            d_x_rot, d_y_rot = self._rotate_data(d_x, d_y)

            # update crack tip position
            crack_tip_x += d_x_rot
            crack_tip_y += d_y_rot

            # plot intermediate results
            if plot_intermediate_results:
                assert folder is not None, 'Please provide a folder to save the plots.'
                assert cd is not None, 'Please provide a CrackDetectionIntercept object.'
                res = {
                    f"$a_{{-1}} = {williams_fit_a_n[-1]:.2f}, b_{{-1}} = {williams_fit_b_n[-1]:.2f}$":
                        [crack_tip_x - self.crack_tip[0], crack_tip_y - self.crack_tip[1]]
                }
                cd.plot(fname=f'iteration_{i}.png', folder=folder, crack_tip_results=res, fmax=self.material.sig_yield)

            if verbose:
                print(f"Iteration {i}: dx = {d_x_rot:+.4f}, dy = {d_y_rot:+.4f}, "
                      f"a_-1 = {williams_fit_a_n[-1]:.4f}, b_-1 = {williams_fit_b_n[-1]:.4f}, "
                      f"a_1 = {williams_fit_a_n[1]:.4f}, b_1 = {williams_fit_b_n[1]:.4f}, "
                      f"crack_tip_corrected = ({crack_tip_x:.4f}, {crack_tip_y:.4f})")

            # log iteration
            williams_dict = {}
            for key in williams_fit_a_n.keys():
                williams_dict[f'a_{key}'] = williams_fit_a_n[key]
            for key in williams_fit_b_n.keys():
                williams_dict[f'b_{key}'] = williams_fit_b_n[key]
            williams_df = pd.DataFrame(williams_dict, index=[i])

            process_log = pd.DataFrame({'Iteration': i, 'dx': d_x_rot, 'dy': d_y_rot,
                                        'crack_tip_x': crack_tip_x,
                                        'crack_tip_y': crack_tip_y}, index=[i])
            step_log = pd.concat([process_log, williams_df], axis=1)
            self.iteration_log = pd.concat([self.iteration_log, step_log], axis=0)

            # stop as soon as correction is smaller than the tolerance
            if np.sqrt(d_x_rot ** 2 + d_y_rot ** 2) < step_tol:
                break

        ct_corr = [crack_tip_x - self.crack_tip[0], crack_tip_y - self.crack_tip[1]]
        print(ct_corr)
        print('------------------------------------')
        return ct_corr
