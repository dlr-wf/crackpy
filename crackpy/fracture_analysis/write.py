import os

import numpy as np

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.crack_tip import unit_of_williams_coefficients


class OutputWriter:
    """Writer class for output of Fracture Analysis Tool.

    Methods:
        * write_header - write header of output file with metadata
        * write_results - write results of fracture analysis

    """

    def __init__(self, path: str, fracture_analysis: FractureAnalysis):
        """Initialize OutputWriter arguments.

        Args:
            path: path to output file
            fracture_analysis: Fracture Analysis data

        """
        self.analysis = fracture_analysis
        self.path = self._make_path(path)
        self.filename = self._set_filename()

    def write_header(self) -> None:
        """Writing a header for the output file."""
        with open(os.path.join(self.path, self.filename), mode='w') as file:
            file.write('############################################################################################\n')
            file.write('#                                                                                          #\n')
            file.write('#                                 Fracture Analysing Tool                                  #\n')
            file.write('#                                                                                          #\n')
            file.write('############################################################################################\n')
            file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write("#############################\n")
            file.write("#     Experimental data     #\n")
            file.write("#############################\n")
            file.write("\n")
            file.write(f'Filename :     {self.filename}\n')
            file.write('\n')
            file.write('<Experiment_data>\n')
            file.write(f'{"Param":>20}, {"Unit":>20}, {"Result":>20} \n')
            file.write(f'{"Crack_tip_x":>20}, {"mm":>20}, {self.analysis.crack_tip.crack_tip_x:20.10f} \n')
            file.write(f'{"Crack_tip_y":>20}, {"mm":>20}, {self.analysis.crack_tip.crack_tip_y:20.10f} \n')
            file.write(f'{"Crack_tip_phi":>20}, {"grad":>20}, {self.analysis.crack_tip.crack_tip_angle:20.10f} \n')
            if self.analysis.data.force is not None:
                file.write(f'{"Force":>20}, {"N":>20}, {self.analysis.data.force:20.10f} \n')
            if self.analysis.data.cycles is not None:
                file.write(f'{"Cycles":>20}, {"1":>20}, {self.analysis.data.cycles:20.10f} \n')
            if self.analysis.data.displacement is not None:
                file.write(f'{"Displacement":>20}, {"mm":>20}, {self.analysis.data.displacement:20.10f} \n')
            if self.analysis.data.potential is not None:
                file.write(f'{"Potential":>20}, {"V":>20}, {self.analysis.data.potential:20.10f} \n')
            if self.analysis.data.cracklength is not None:
                file.write(f'{"Cracklength_dcpd":>20}, {"mm":>20}, {self.analysis.data.cracklength:20.10f} \n')
            if self.analysis.data.time is not None:
                file.write(f'{"timestamp":>20}, {"s":>20}, {self.analysis.data.time:20.10f} \n')
            file.write('</Experiment_data>\n')
            file.write('\n')

    def write_results(self) -> None:
        """Write results of fracture analysis into output file."""
        with open(os.path.join(self.path, self.filename), 'a') as file:

            if self.analysis.optimization_properties is not None:

                file.write('\n')
                file.write("#############################\n")
                file.write("#         CJP model         #\n")
                file.write("#############################\n")
                file.write('\n')
                file.write('<CJP_results>\n')
                file.write(f'{"Param":>10}, {"Unit":>20}, {"Result":>20} \n')
                file.write(f'{"Error":>10}, {"1":>20}, {self.analysis.res_cjp["Error"]:20.10f} \n')
                file.write(f'{"K_F":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.res_cjp["K_F"]:20.10f} \n')
                file.write(f'{"K_R":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.res_cjp["K_R"]:20.10f} \n')
                file.write(f'{"K_S":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.res_cjp["K_S"]:20.10f} \n')
                file.write(f'{"K_II":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.res_cjp["K_II"]:20.10f} \n')
                file.write(f'{"T":>10}, {"MPa":>20}, {self.analysis.res_cjp["T"]:20.10f} \n')
                file.write('</CJP_results>\n')
                file.write('\n')

                file.write("#############################\n")
                file.write("#      Williams fitting     #\n")
                file.write("#############################\n")
                file.write('\n')
                file.write('<Williams_fit_results>\n')
                file.write(f'{"Param":>10}, {"Unit":>20}, {"Result":>20} \n')
                file.write(f'{"Error":>10}, {"1":>20}, {self.analysis.sifs_fit["Error"]:20.10f} \n')
                file.write(f'{"K_I":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.sifs_fit["K_I"]:20.10f} \n')
                file.write(f'{"K_II":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.sifs_fit["K_II"]:20.10f} \n')
                file.write(f'{"T":>10}, {"MPa":>20}, {self.analysis.sifs_fit["T"]:20.10f} \n')
                for n, a in self.analysis.williams_fit_a_n.items():
                    file.write(f'{f"a_{n}":>10}, {unit_of_williams_coefficients(n):>20}, {a:20.10f} \n')
                for n, b in self.analysis.williams_fit_b_n.items():
                    file.write(f'{f"b_{n}":>10}, {unit_of_williams_coefficients(n):>20}, {b:20.10f} \n')
                file.write('</Williams_fit_results>\n')
                file.write('\n')

            if self.analysis.integral_properties is not None:
                file.write("###################################\n")
                file.write("#    SIFs integral evaluation     #\n")
                file.write("###################################\n")
                file.write('\n')
                file.write('<SIFs_integral>\n')

                file.write(f'{"Param":>20}, {"Unit":>20}, {"Mean":>20}, {"Median":>20}, {"Mean_wo_outliers":>20} \n')
                file.write(
                    f'{"J":>20}, {"N/mm":>20}, '
                    f'{self.analysis.sifs_int["mean"]["j"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["j"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["j"]:20.10f}\n')
                file.write(
                    f'{"K_J":>20}, {"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["sif_j"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["sif_j"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["sif_j"]:20.10f}\n')
                file.write(
                    f'{"K_I_interac":>20}, '
                    f'{"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["sif_k_i"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["sif_k_i"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["sif_k_i"]:20.10f}\n')
                file.write(
                    f'{"K_II_interac":>20}, '
                    f'{"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["sif_k_ii"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["sif_k_ii"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["sif_k_ii"]:20.10f}\n')
                file.write(
                    f'{"T_interac":>20}, {"MPa":>20}, '
                    f'{self.analysis.sifs_int["mean"]["t_stress_int"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["t_stress_int"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["t_stress_int"]:20.10f}\n')
                file.write(
                    f'{"K_I_Chen":>20}, '
                    f'{"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["k_i_chen"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["k_i_chen"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["k_i_chen"]:20.10f}\n')
                file.write(
                    f'{"K_II_Chen":>20}, '
                    f'{"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["k_ii_chen"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["k_ii_chen"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["k_ii_chen"]:20.10f}\n')
                file.write(
                    f'{"T_Chen":>20}, '
                    f'{"MPa":>20}, '
                    f'{self.analysis.sifs_int["mean"]["t_stress_chen"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["t_stress_chen"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["t_stress_chen"]:20.10f}\n')
                file.write(
                    f'{"T_SDM":>20}, '
                    f'{"MPa":>20}, '
                    f'{self.analysis.sifs_int["mean"]["t_stress_sdm"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["t_stress_sdm"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["t_stress_sdm"]:20.10f}\n')

                file.write('</SIFs_integral>\n')
                file.write("\n")
                file.write("\n")

                if self.analysis.integral_properties.buckner_williams_terms is not None:
                    file.write("#################################\n")
                    file.write("#    Bueckner-Chen integral     #\n")
                    file.write("#################################\n")
                    file.write('\n')
                    file.write('<Bueckner_Chen_integral>\n')

                    file.write(
                        f'{"Param":>10}, {"Unit":>20}, {"Mean":>20}, {"Median":>20}, {"Mean_wo_outliers":>20} \n')

                    terms = self.analysis.williams_int[0, :, 0]
                    for term_index, term in enumerate(terms):
                        file.write(
                            f'{f"a_{term:.0f}":>10}, '
                            f'{unit_of_williams_coefficients(term):>20}, '
                            f'{self.analysis.sifs_int["mean"]["williams_int_a_n"][term_index]:20.10f}, '
                            f'{self.analysis.sifs_int["median"]["williams_int_a_n"][term_index]:20.10f}, '
                            f'{self.analysis.sifs_int["rej_out_mean"]["williams_int_a_n"][term_index]:20.10f}\n')

                    for term_index, term in enumerate(terms):
                        file.write(
                            f'{f"b_{term:.0f}":>10}, '
                            f'{unit_of_williams_coefficients(term):>20}, '
                            f'{self.analysis.sifs_int["mean"]["williams_int_b_n"][term_index]:20.10f}, '
                            f'{self.analysis.sifs_int["median"]["williams_int_b_n"][term_index]:20.10f}, '
                            f'{self.analysis.sifs_int["rej_out_mean"]["williams_int_b_n"][term_index]:20.10f}\n')

                    file.write('</Bueckner_Chen_integral>\n')
                    file.write('\n')
                    file.write('\n')

                file.write("#############################\n")
                file.write("#    Integral Path SIFs     #\n")
                file.write("#############################\n")
                file.write('\n')
                file.write('<Path_SIFs>\n')
                file.write(
                    f'{"J [N/mm]":>20}, '
                    f'{"K_J [MPa*sqrt(m)]":>20}, '
                    f'{"K_I [MPa*sqrt(m)]":>20}, '
                    f'{"K_II [MPa*sqrt(m)]":>20}, '
                    f'{"T_Chen [MPa]":>20}, '
                    f'{"T_SDM [MPa]":>20}, '
                    f'{"T_Int [MPa]":>20} \n')
                for each_path_index in range(len(self.analysis.results)):
                    file.write(
                        f'{self.analysis.results[each_path_index][0]:20.10f}, '
                        f'{self.analysis.results[each_path_index][1]:20.10f}, '
                        f'{self.analysis.results[each_path_index][2]:20.10f}, '
                        f'{self.analysis.results[each_path_index][3]:20.10f}, '
                        f'{self.analysis.results[each_path_index][4]:20.10f}, '
                        f'{self.analysis.results[each_path_index][5]:20.10f}, '
                        f'{self.analysis.results[each_path_index][6]:20.10f} \n')
                file.write('</Path_SIFs>\n')
                file.write('\n')
                file.write("\n")

                file.write("#############################\n")
                file.write("#  Integral Path Williams   #\n")
                file.write("#############################\n")
                file.write('\n')
                file.write('<Path_Williams_a_n>\n')

                for i, term in enumerate(terms):
                    string = f'a_{term:.0f} [{unit_of_williams_coefficients(term)}]'
                    if i == len(terms) - 1:
                        file.write(f'{string:>25s}')
                    else:
                        file.write(f'{string:>25s},')
                file.write('\n')
                for each_path in self.analysis.williams_int_a_n:
                    for i, each_term in enumerate(each_path):
                        if i == len(each_path) - 1:
                            file.write(f'{each_term:>25.10f}')
                        else:
                            file.write(f'{each_term:>25.10f},')
                    file.write('\n')

                file.write('</Path_Williams_a_n>\n')
                file.write("\n")
                file.write('<Path_Williams_b_n>\n')

                for i, term in enumerate(terms):
                    string = f'b_{term:.0f} [{unit_of_williams_coefficients(term)}]'
                    if i == len(terms) - 1:
                        file.write(f'{string:>25s}')
                    else:
                        file.write(f'{string:>25s},')
                file.write('\n')
                for each_path in self.analysis.williams_int_b_n:
                    for i, each_term in enumerate(each_path):
                        if i == len(each_path) - 1:
                            file.write(f'{each_term:>25.10f}')
                        else:
                            file.write(f'{each_term:>25.10f},')
                    file.write('\n')

                file.write('</Path_Williams_b_n>\n')
                file.write("\n")
                file.write("\n")

                file.write("#############################\n")
                file.write("#      Path properties      #\n")
                file.write("#############################\n")
                file.write('\n')
                file.write('<Path_Properties>\n')
                file.write(
                    f'{"NumOfNodes":>10}, '
                    f'{"TickSize[mm]":>12}, '
                    f'{"LineXL":>6}, '
                    f'{"LineXR":>6}, '
                    f'{"LineYB":>6}, '
                    f'{"LineYT":>6}, '
                    f'{"TopOffset":>9}, '
                    f'{"BotOffset":>9} \n')
                for each_path_index in range(len(self.analysis.results)):
                    file.write(
                        f'{self.analysis.num_of_path_nodes[each_path_index]:10.0f}, '
                        f'{self.analysis.tick_sizes[each_path_index]:12.4f}, '
                        f'{self.analysis.int_sizes[each_path_index][0]:6.2f}, '
                        f'{self.analysis.int_sizes[each_path_index][1]:6.2f}, '
                        f'{self.analysis.int_sizes[each_path_index][2]:6.2f}, '
                        f'{self.analysis.int_sizes[each_path_index][3]:6.2f}, '
                        f'{self.analysis.integration_points[each_path_index][1][-1]:9.2f}, '
                        f'{self.analysis.integration_points[each_path_index][1][0]:9.2f} \n')
                file.write('</Path_Properties>\n')
                file.write('\n\n\n')

    @staticmethod
    def _make_path(output_path) -> str:
        """Create and return path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def _set_filename(self) -> str:
        """Transforms 'Filename.txt' -> 'Filename_right_Output.txt'"""
        return os.path.split(self.analysis.nodemap_file)[-1][:-4] + '_' + self.analysis.crack_tip.left_or_right + '_Output.txt'
