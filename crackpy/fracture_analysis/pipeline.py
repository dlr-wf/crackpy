import os

import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from rich import progress as progress_rich
import numpy as np
import pandas as pd

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.data_processing import CrackTipInfo, InputData
from crackpy.fracture_analysis.optimization import OptimizationProperties
from crackpy.fracture_analysis.line_integration import IntegralProperties
from crackpy.fracture_analysis.plot import PlotSettings, Plotter
from crackpy.fracture_analysis.write import OutputWriter
from crackpy.structure_elements.data_files import NodemapStructure, Nodemap
from crackpy.structure_elements.material import Material


def single_run(
        index: int,
        data,
        material: Material,
        nodemap_path: str,
        nodemap_structure: NodemapStructure,
        integral_props_by_nodemap: dict,
        opt_props: OptimizationProperties,
        output_path: str,
        plot_sets: PlotSettings or None,
        prog,
        task_id
):
    """Run fracture analysis of a single nodemap.

    Args:
        index: running index of data
        data: input data as data frame
        material: obj of class Material
        nodemap_path: path of nodemap
        nodemap_structure: data structure of nodemap
        integral_props_by_nodemap: dictionary of the line integral path properties with *index* as keys
                                   If nodemap is not in dictionary, the integral evaluation is skipped.
        opt_props: obj of class OptimizationProperties
        output_path: path where the plots and results are saved
        plot_sets: settings for plotting the results
        prog: progress bar
        task_id: task id of progress bar

    """
    # get crack tip info from data
    crack_tip = CrackTipInfo(
        crack_tip_x=data['Crack Tip x [mm]'],
        crack_tip_y=data['Crack Tip y [mm]'],
        crack_tip_angle=data['Crack Angle'],
        left_or_right=data['Side']
    )

    # import and transform data
    nodemap = Nodemap(data['Filename'], nodemap_path, structure=nodemap_structure)
    input_data = InputData(nodemap)
    input_data.calc_stresses(material)
    input_data.transform_data(crack_tip.crack_tip_x, crack_tip.crack_tip_y, crack_tip.crack_tip_angle)

    # set integral properties
    if index in integral_props_by_nodemap.keys():
        int_props = integral_props_by_nodemap[index]
    else:
        int_props = None

    # run fracture analysis
    analysis = FractureAnalysis(
        material=material,
        nodemap=nodemap,
        data=input_data,
        crack_tip_info=crack_tip,
        integral_properties=int_props,
        optimization_properties=opt_props
    )
    analysis.run(prog, task_id)

    # write output to txt file
    writer = OutputWriter(path=os.path.join(output_path, 'txt-files'), fracture_analysis=analysis)
    writer.write_header()
    writer.write_results()

    # plot paths and results_df
    if plot_sets is not None:
        plotter = Plotter(path=os.path.join(output_path, 'plots'), fracture_analysis=analysis, plot_sets=plot_sets)
        plotter.plot()


class FractureAnalysisPipeline:
    """Fracture Analysis Pipeline taking input file with necessary crack tip information, plot settings,
    optimization properties, and nodemap structure as input.

    The pipeline runs the fracture analysis for each nodemap in the input file. The results are saved in txt-files.
    If plotting is enabled, the results are plotted and saved in the 'plots' subfolder.

    The pipeline is able to calculate

    - J-integral
    - K_I and K_II with the interaction integral
    - T-stress with the interaction integral
    - higher-order terms (HOSTs and HORTs) w/ fitting method (ODM)
    - K_F, K_R, K_S, K_II and T w/ the CJP model
    - (BETA) T-stress with the Bueckner-Chen integral
    - (BETA) higher-order terms (HOSTs and HORTs) w/ Bueckner-integral

    Methods:
        * find_integral_props - finds the integral properties at max load automatically
        * set_integral_props_manually - manually set global integral properties for all nodemaps
        * run - run pipeline to create plots and output files
        * multi_run - run pipeline on multiple processes to save time

    """

    def __init__(
            self,
            material: Material,
            nodemap_path: str,
            input_file: str,
            output_path: str,
            plot_sets: PlotSettings = None,
            optimization_properties: OptimizationProperties = None,
            integral_properties: IntegralProperties = None,
            nodemap_structure: NodemapStructure = NodemapStructure()
    ):
        """Initialize pipeline.

        Args:
            material: obj of class Material
            nodemap_path: nodemap data path
            input_file: path to input file
            output_path: base path for output of pipeline with subfolders 'plots' and 'txt-files'
            plot_sets: obj of class PlotSettings (If None, no plots will be created)
            optimization_properties: obj of class OptimizationProperties (If None, fitting methods are skipped)
            integral_properties: obj of class IntegralProperties (If None, integral evaluation is skipped)
            nodemap_structure: obj of class NodemapStructure (If not specified, the default structure is used)

        """
        self.material = material
        self.nodemap_path = nodemap_path
        self.nodemap_structure = nodemap_structure
        self.input_file = input_file
        self.input_df = self._get_df_from_file()
        self.plot_sets = plot_sets
        self.opt_props = optimization_properties
        self.output_path = self._make_path(output_path)

        print("\n\nRun fracture analysis pipeline.")

        # user warnings
        if self.plot_sets is None:
            warnings.warn("Plotting of outputs is turned off."
                          " If you want to plot the pipeline's outputs, use the 'plot_sets' argument.")
        if self.opt_props is None:
            warnings.warn("Fitting methods are turned off."
                          " If you want to use fitting methods, use the 'optimization_properties' argument.")
        if integral_properties is None:
            warnings.warn("Integral evaluation is turned off."
                          " If you want to evaluate integrals, use the 'integral_properties' argument "
                          "or the method 'find_integral_props'.")

        # initialize stages to max force stages for storage in dictionary
        self.stages_to_max_force_stages = None

        # initialize dictionary of integral properties
        self.integral_props = {}
        if integral_properties is not None:
            self.set_integral_props_manually(integral_properties)

    def set_integral_props_manually(self, int_props: IntegralProperties):
        """Set one integral path property for all nodemaps in input_df.

        Args:
            int_props: one integral property that is set for all nodemaps of the pipeline

        """
        self.integral_props = {index: int_props for index, _ in self.input_df.iterrows()}

    def find_max_force_stages(self, max_force: float, tol: float = 20) -> dict:
        """Find stages of maximal force and store as dictionary stages_to_max_force_stages.

        Args:
            max_force: maximal force of the pipeline data
            tol: tolerance for finding the stages

        Returns:
            stages_to_max_force_stages: dictionary of stages to max force stages

        """
        filtered_stages = []
        max_force_cycles_to_stages = {}
        stages_to_cycles = {}

        # find stages of maximal force
        for _, data in self.input_df.iterrows():
            stage = int(data["Filename"].split("_")[-1].split(".")[0])
            nodemap = Nodemap(name=data["Filename"], folder=self.nodemap_path, structure=self.nodemap_structure)
            data = InputData()
            data.set_data_file(os.path.join(nodemap.folder, nodemap.name))
            data.read_header()

            if data.force is None:
                # no force -> add to filtered_stages
                filtered_stages.append(stage)
            if data.force > max_force - tol:
                filtered_stages.append(stage)
                # get cycles to stages dictionaries
                if data.cycles is not None:
                    max_force_cycles_to_stages[data.cycles] = stage
            if data.cycles is not None:
                stages_to_cycles[stage] = data.cycles

        # assign stages to their corresponding max force stage
        stages_to_max_force_stages = {}
        for stage, cycle in stages_to_cycles.items():
            max_force_cycles_array = np.asarray(list(max_force_cycles_to_stages.keys()))
            closest_max_force_cycle = np.argmin(np.abs(max_force_cycles_array - cycle))
            closest_max_force_stage = max_force_cycles_to_stages[max_force_cycles_array[closest_max_force_cycle]]
            stages_to_max_force_stages[stage] = closest_max_force_stage
        self.stages_to_max_force_stages = stages_to_max_force_stages

        return self.stages_to_max_force_stages

    def find_integral_props(self, stages_to_max_force_stages: dict or None = None):
        """Find the integral properties at maximal load automatically
        and assign these also to the subsequent smaller load steps.

        Args:
            stages_to_max_force_stages: dictionary of stages to max force stages

        """
        print("\n\nFind integral properties...")

        # warn the user that this function is a BETA version
        warnings.warn("The method 'find_integral_props' is a BETA version."
                      " Please check that detected integral paths are correct.")

        if stages_to_max_force_stages is None:
            stages_to_max_force_stages = self.stages_to_max_force_stages
        if stages_to_max_force_stages is None:
            raise ValueError("Missing 'stages_to_max_force_stages'. Run method 'find_max_force_stages' first.")

        index_to_side = {}
        index_to_stage = {}
        side_to_stage_to_index = {'left': {}, 'right': {}}
        for index, data in self.input_df.iterrows():
            print(f'\r Progress... {index + 1}/{len(self.input_df)}', end='')

            stage = int(data["Filename"].split("_")[-1].split(".")[0])
            side = data["Side"]
            index_to_stage[index] = stage
            index_to_side[index] = side
            side_to_stage_to_index[side][stage] = index

            # only consider max force stages
            if stage not in stages_to_max_force_stages.values():
                continue

            # get crack tip info from data
            crack_tip = CrackTipInfo(
                crack_tip_x=data['Crack Tip x [mm]'],
                crack_tip_y=data['Crack Tip y [mm]'],
                crack_tip_angle=data['Crack Angle'],
                left_or_right=data['Side']
            )

            # transform input data
            input_nodemap = Nodemap(name=data["Filename"], folder=self.nodemap_path, structure=self.nodemap_structure)
            input_data = InputData(input_nodemap)
            input_data.calc_stresses(self.material)
            input_data.transform_data(crack_tip.crack_tip_x, crack_tip.crack_tip_y, crack_tip.crack_tip_angle)

            # get integral properties
            if index in self.integral_props:
                integral_properties = self.integral_props[index]
            else:
                integral_properties = IntegralProperties()
            try:
                integral_properties.set_automatically(input_data, auto_detect_threshold=self.material.sig_yield)
            except ValueError:
                print(f'Could not find integral properties automatically for stage {stage}.')
            self.integral_props[index] = integral_properties

        # assign integral properties to missing stages
        for index, data in self.input_df.iterrows():
            if index not in self.integral_props.keys():
                stage = index_to_stage[index]
                side = index_to_side[index]
                max_force_stage = stages_to_max_force_stages[stage]
                corr_index = side_to_stage_to_index[side][max_force_stage]
                self.integral_props[index] = self.integral_props[corr_index]

    def run(self, num_of_kernels: int = 1):
        """Run fracture analysis pipeline. This method is the main method of the pipeline.

        Args:
            num_of_kernels: number of subprocesses, e.g. number of kernels used
                            If None, then (#CPUs / 2) processes are used

        """
        # max number of processes is half of the number of CPUs
        num_of_kernels = min(multiprocessing.cpu_count() // 2, num_of_kernels)

        with progress_rich.Progress(
                "[progress.description]{task.description}",
                progress_rich.BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                progress_rich.TimeRemainingColumn(),
                progress_rich.TimeElapsedColumn()
        ) as progress:
            futures = []  # keep track of the jobs
            with multiprocessing.Manager() as manager:
                _progress = manager.dict()
                overall_progress_task = progress.add_task("[green]Overall fracture analysis progress:")

                with ProcessPoolExecutor(max_workers=num_of_kernels) as executor:

                    # submit jobs / nodemaps
                    for index, data in self.input_df.iterrows():
                        task_id = progress.add_task(f"{data['Filename']}:", visible=False)
                        futures.append(executor.submit(single_run, index, data, self.material, self.nodemap_path,
                                                       self.nodemap_structure, self.integral_props, self.opt_props,
                                                       self.output_path, self.plot_sets, _progress, task_id))

                    # monitor the progress
                    while sum([future.done() for future in futures]) < len(futures):
                        n_finished = sum([future.done() for future in futures])
                        progress.update(overall_progress_task, completed=n_finished, total=len(futures))
                        for task_id, update_data in _progress.items():
                            latest = update_data["progress"]
                            total = update_data["total"]
                            # update the progress bar for this task
                            progress.update(task_id, completed=latest, total=total, visible=latest < total)
                    progress.update(overall_progress_task, completed=n_finished, total=len(futures))

                    # raise any errors
                    for future in futures:
                        future.result()

    def _get_df_from_file(self):
        """Read and return data frame from input file."""
        df = pd.read_csv(self.input_file, sep=",", skipinitialspace=True)
        return df

    @staticmethod
    def _make_path(output_path):
        """Create and return output path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path
