from pathlib import Path
import json
import numpy as np
import datetime
import logging

from crackpy.fracture_analysis.analysis import FractureAnalysis
from crackpy.fracture_analysis.crack_tip import unit_of_williams_coefficients

logger = logging.getLogger(__name__)


class OutputWriter:
    """Writer class for output of Fracture Analysis Tool.

    Attributes:
        analysis: FractureAnalysis instance
        path: Output path
        filename: Output filename
        json_path: JSON output path

    Methods:
        * write_header - write header of output file with metadata
        * write_results - write results of fracture analysis
        * write_json - write results of fracture analysis into json file

    """

    def __init__(self, path: str | Path, fracture_analysis: FractureAnalysis) -> None:
        """Initialize OutputWriter arguments.

        Args:
            path: path to output file
            fracture_analysis: Fracture Analysis data

        """
        self.analysis = fracture_analysis
        self.path = self._make_path(path)
        self.filename = self._set_filename()
        self.json_path = None

        logger.debug("OutputWriter initialized: %s in %s", self.filename, self.path)

    def write_header(self) -> None:
        """Writing a header for the output file."""
        out_file = Path(self.path) / self.filename
        logger.debug("Writing header to output file: %s", out_file)

        with open(out_file, mode='w') as file:
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
        out_file = Path(self.path) / self.filename
        logger.debug(f"Writing fracture analysis results to: {out_file}")

        with open(out_file, 'a') as file:

            if self.analysis.optimization_properties is not None:
                logger.debug("Writing CJP and Williams fitting results")
                file.write('\n')
                file.write("#######################################\n")
                file.write("#     CJP model (Mode I / Mode II)    #\n")
                file.write("#######################################\n")
                file.write('\n')
                file.write('<CJP_results>\n')
                file.write(f'{"Param":>10}, {"Unit":>20}, {"Result":>20} \n')
                file.write(f'{"Error":>10}, {"1":>20}, {self.analysis.cjp_res_mm["Error"]:20.10f} \n')
                file.write(f'{"K_F":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.cjp_res_mm["K_F"]:20.10f} \n')
                file.write(f'{"K_R":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.cjp_res_mm["K_R"]:20.10f} \n')
                file.write(f'{"K_S":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.cjp_res_mm["K_S"]:20.10f} \n')
                file.write(f'{"K_II":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.cjp_res_mm["K_II"]:20.10f} \n')
                file.write(f'{"T":>10}, {"MPa":>20}, {self.analysis.cjp_res_mm["T"]:20.10f} \n')
                file.write('</CJP_results>\n')
                file.write('\n')
                file.write("##############################\n")
                file.write("#     CJP model (Mode I)     #\n")
                file.write("##############################\n")
                file.write("\n")
                file.write('<CJP_modeI_results>\n')
                file.write(f'{"Param":>10}, {"Unit":>20}, {"Result":>20} \n')
                file.write(f'{"Error":>10}, {"1":>20}, {self.analysis.cjp_res_m1["Error"]:20.10f} \n')
                file.write(f'{"K_F":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.cjp_res_m1["K_F"]:20.10f} \n')
                file.write(f'{"K_R":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.cjp_res_m1["K_R"]:20.10f} \n')
                file.write(f'{"K_S":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.cjp_res_m1["K_S"]:20.10f} \n')
                file.write(f'{"T_x":>10}, {"MPa":>20}, {self.analysis.cjp_res_m1["T_x"]:20.10f} \n')
                file.write(f'{"T_y":>10}, {"MPa":>20}, {self.analysis.cjp_res_m1["T_y"]:20.10f} \n')
                file.write('</CJP_modeI_results>\n')
                file.write("#############################\n")
                file.write("#      Williams fitting     #\n")
                file.write("#############################\n")
                file.write('\n')
                file.write('<Williams_fit_results>\n')
                file.write(f'{"Param":>10}, {"Unit":>20}, {"Result":>20} \n')
                file.write(f'{"Error_xy":>10}, {"1":>20}, {self.analysis.williams_fit_res["Error_xy"]:20.10f} \n')
                file.write(f'{"Error_z":>10}, {"1":>20}, {self.analysis.williams_fit_res["Error_z"]:20.10f} \n')
                file.write(f'{"K_I":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.williams_fit_res["K_I"]:20.10f} \n')
                file.write(f'{"K_II":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.williams_fit_res["K_II"]:20.10f} \n')
                file.write(f'{"K_III":>10}, {"MPa*m^{1/2}":>20}, {self.analysis.williams_fit_res["K_III"]:20.10f} \n')
                file.write(f'{"T":>10}, {"MPa":>20}, {self.analysis.williams_fit_res["T"]:20.10f} \n')
                for n, a in self.analysis.williams_fit_a_n.items():
                    file.write(f'{f"a_{n}":>10}, {unit_of_williams_coefficients(n):>20}, {a:20.10f} \n')
                for n, b in self.analysis.williams_fit_b_n.items():
                    file.write(f'{f"b_{n}":>10}, {unit_of_williams_coefficients(n):>20}, {b:20.10f} \n')
                for n, c in self.analysis.williams_fit_c_n.items():
                    file.write(f'{f"c_{n}":>10}, {unit_of_williams_coefficients(n):>20}, {c:20.10f} \n')
                file.write('</Williams_fit_results>\n')
                file.write('\n')

            if self.analysis.integral_properties is not None:
                logger.debug("Writing integral evaluation results")
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
                    f'{"J_I":>20}, {"N/mm":>20}, '
                    f'{self.analysis.sifs_int["mean"]["decomp_j_1"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["decomp_j_1"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["decomp_j_1"]:20.10f}\n')
                file.write(
                    f'{"J_II":>20}, {"N/mm":>20}, '
                    f'{self.analysis.sifs_int["mean"]["decomp_j_2"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["decomp_j_2"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["decomp_j_2"]:20.10f}\n')
                file.write(
                    f'{"J_III":>20}, {"N/mm":>20}, '
                    f'{self.analysis.sifs_int["mean"]["decomp_j_3"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["decomp_j_3"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["decomp_j_3"]:20.10f}\n')
                file.write(
                    f'{"K_I_J":>20}, {"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["decomp_K_1"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["decomp_K_1"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["decomp_K_1"]:20.10f}\n')
                file.write(
                    f'{"K_II_J":>20}, {"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["decomp_K_2"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["decomp_K_2"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["decomp_K_2"]:20.10f}\n')
                file.write(
                    f'{"K_III_J":>20}, {"MPa*m^{1/2}":>20}, '
                    f'{self.analysis.sifs_int["mean"]["decomp_K_3"]:20.10f}, '
                    f'{self.analysis.sifs_int["median"]["decomp_K_3"]:20.10f}, '
                    f'{self.analysis.sifs_int["rej_out_mean"]["decomp_K_3"]:20.10f}\n')
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
                    file.write("##################################\n")
                    file.write("#     Bueckner-Chen integral     #\n")
                    file.write("##################################\n")
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
                    f'{"T_Int [MPa]":>20},'
                    f'{"J_1 [N/mm]":>20},'
                    f'{"J_2 [N/mm]":>20},'
                    f'{"J_3 [N/mm]":>20},'
                    f'{"K_J-I [N/mm]":>20},'
                    f'{"K_J-II [N/mm]":>20},'
                    f'{"K_J-III [N/mm]":>20}\n')
                for each_path_index in range(len(self.analysis.path_results)):
                    file.write(
                        f'{self.analysis.path_results[each_path_index][0]:20.10f}, '
                        f'{self.analysis.path_results[each_path_index][1]:20.10f}, '
                        f'{self.analysis.path_results[each_path_index][2]:20.10f}, '
                        f'{self.analysis.path_results[each_path_index][3]:20.10f}, '
                        f'{self.analysis.path_results[each_path_index][4]:20.10f}, '
                        f'{self.analysis.path_results[each_path_index][5]:20.10f}, '
                        f'{self.analysis.path_results[each_path_index][6]:20.10f},'
                        f'{self.analysis.path_results[each_path_index][7]:20.10f},'
                        f'{self.analysis.path_results[each_path_index][8]:20.10f},'
                        f'{self.analysis.path_results[each_path_index][9]:20.10f},'
                        f'{self.analysis.path_results[each_path_index][10]:20.10f},'
                        f'{self.analysis.path_results[each_path_index][11]:20.10f},'
                        f'{self.analysis.path_results[each_path_index][12]:20.10f}\n')
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
                for each_path_index in range(len(self.analysis.path_results)):
                    file.write(
                        f'{self.analysis.num_of_path_nodes[each_path_index]:10.0f}, '
                        f'{self.analysis.tick_sizes[each_path_index]:12.4f}, '
                        f'{self.analysis.path_sizes[each_path_index][0]:6.2f}, '
                        f'{self.analysis.path_sizes[each_path_index][1]:6.2f}, '
                        f'{self.analysis.path_sizes[each_path_index][2]:6.2f}, '
                        f'{self.analysis.path_sizes[each_path_index][3]:6.2f}, '
                        f'{self.analysis.integration_points[each_path_index][1][-1]:9.2f}, '
                        f'{self.analysis.integration_points[each_path_index][1][0]:9.2f} \n')
                file.write('</Path_Properties>\n')
                file.write('\n\n\n')

    def write_json(self, path=None):
        """Write results of fracture analysis into json file.

        Args:
            path: Optional path to json file. If None, the path from the OutputWriter class is used.

        """
        logger.debug("Writing JSON results to path: %s", (path if path else self.path))

        if path is None:
            self.json_path = self.path
        else:
            self.json_path = self._make_path(path)


        json_dict = {'filename': self.filename,
                     'evaluation_UTC_datetime': datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
                     'experiment_data': {}}
        json_dict['experiment_data']['crack_tip_x'] = {"unit": "mm",
                                                       "result": self.analysis.crack_tip.crack_tip_x}
        json_dict['experiment_data']['crack_tip_y'] = {"unit": "mm",
                                                       "result": self.analysis.crack_tip.crack_tip_y}
        json_dict['experiment_data']['crack_tip_phi'] = {"unit": "grad",
                                                         "result": self.analysis.crack_tip.crack_tip_angle}

        if self.analysis.data.force is not None:
            json_dict['experiment_data']['force'] = {"unit": "N",
                                                     "result": self.analysis.data.force}
        if self.analysis.data.cycles is not None:
            json_dict['experiment_data']['cycles'] = {"unit": "1",
                                                      "result": self.analysis.data.cycles}
        if self.analysis.data.displacement is not None:
            json_dict['experiment_data']['displacement'] = {"unit": "mm",
                                                            "result": self.analysis.data.displacement}
        if self.analysis.data.potential is not None:
            json_dict['experiment_data']['potential'] = {"unit": "V",
                                                         "result": self.analysis.data.potential}
        if self.analysis.data.cracklength is not None:
            json_dict['experiment_data']['cracklength_dcpd'] = {"unit": "mm",
                                                                "result": self.analysis.data.cracklength}
        if self.analysis.data.time is not None:
            json_dict['experiment_data']['timestamp'] = {"unit": "s",
                                                         "result": self.analysis.data.time}


        if self.analysis.optimization_properties is not None:
            json_dict['CJP_results'] = {}
            json_dict['CJP_results']['error'] = {"unit": "1",
                                                 "result": self.analysis.cjp_res_mm["Error"]}
            json_dict['CJP_results']['K_F'] = {"unit": "MPa*m^{1/2}",
                                               "result": self.analysis.cjp_res_mm["K_F"]}
            json_dict['CJP_results']['K_R'] = {"unit": "MPa*m^{1/2}",
                                               "result": self.analysis.cjp_res_mm["K_R"]}
            json_dict['CJP_results']['K_S'] = {"unit": "MPa*m^{1/2}",
                                               "result": self.analysis.cjp_res_mm["K_S"]}
            json_dict['CJP_results']['K_II'] = {"unit": "MPa*m^{1/2}",
                                                "result": self.analysis.cjp_res_mm["K_II"]}
            json_dict['CJP_results']['T'] = {"unit": "MPa",
                                             "result": self.analysis.cjp_res_mm["T"]}

            json_dict['CJP_modeI_results'] = {}
            json_dict['CJP_modeI_results']['error'] = {"unit": "1",
                                                       "result": self.analysis.cjp_res_m1["Error"]}
            json_dict['CJP_modeI_results']['K_F'] = {"unit": "MPa*m^{1/2}",
                                                     "result": self.analysis.cjp_res_m1["K_F"]}
            json_dict['CJP_modeI_results']['K_R'] = {"unit": "MPa*m^{1/2}",
                                                     "result": self.analysis.cjp_res_m1["K_R"]}
            json_dict['CJP_modeI_results']['K_S'] = {"unit": "MPa*m^{1/2}",
                                                     "result": self.analysis.cjp_res_m1["K_S"]}
            json_dict['CJP_modeI_results']['T_x'] = {"unit": "MPa",
                                                     "result": self.analysis.cjp_res_m1["T_x"]}
            json_dict['CJP_modeI_results']['T_y'] = {"unit": "MPa",
                                                     "result": self.analysis.cjp_res_m1["T_y"]}

            json_dict['Williams_fit_results'] = {}
            json_dict['Williams_fit_results']['error_xy'] = {"unit": "1",
                                                          "result": self.analysis.williams_fit_res["Error_xy"]}
            json_dict['Williams_fit_results']['error_z'] = {"unit": "1",
                                                         "result": self.analysis.williams_fit_res["Error_z"]}
            json_dict['Williams_fit_results']['K_I'] = {"unit": "MPa*m^{1/2}",
                                                        "result": self.analysis.williams_fit_res["K_I"]}
            json_dict['Williams_fit_results']['K_II'] = {"unit": "MPa*m^{1/2}",
                                                         "result": self.analysis.williams_fit_res["K_II"]}
            json_dict['Williams_fit_results']['K_III'] = {"unit": "MPa*m^{1/2}",
                                                          "result": self.analysis.williams_fit_res["K_III"]}

            json_dict['Williams_fit_results']['T'] = {"unit": "MPa",
                                                      "result": self.analysis.williams_fit_res['T']}
            for n, a in self.analysis.williams_fit_a_n.items():
                json_dict['Williams_fit_results'][f'a_{n}'] = {"unit": unit_of_williams_coefficients(n),
                                                               "result": a}
            for n, b in self.analysis.williams_fit_b_n.items():
                json_dict['Williams_fit_results'][f'b_{n}'] = {"unit": unit_of_williams_coefficients(n),
                                                               "result": b}
            for n, c in self.analysis.williams_fit_c_n.items():
                json_dict['Williams_fit_results'][f'c_{n}'] = {"unit": unit_of_williams_coefficients(n),
                                                               "result": c}

        if self.analysis.integral_properties is not None:
            json_dict['SIFs_integral'] = {}
            json_dict['SIFs_integral']['J'] = {"unit": "N/mm",
                                               "mean": self.analysis.sifs_int["mean"]["j"],
                                               "median": self.analysis.sifs_int["median"]["j"],
                                               "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"]["j"]}
            json_dict['SIFs_integral']['K_J'] = {"unit": "MPa*m^{1/2}",
                                                 "mean": self.analysis.sifs_int["mean"]["sif_j"],
                                                 "median": self.analysis.sifs_int["median"]["sif_j"],
                                                 "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"]["sif_j"]}
            json_dict['SIFs_integral']['K_I_interac'] = {"unit": "MPa*m^{1/2}",
                                                         "mean": self.analysis.sifs_int["mean"]["sif_k_i"],
                                                         "median": self.analysis.sifs_int["median"]["sif_k_i"],
                                                         "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                             "sif_k_i"]}
            json_dict['SIFs_integral']['K_II_interac'] = {"unit": "MPa*m^{1/2}",
                                                          "mean": self.analysis.sifs_int["mean"]["sif_k_ii"],
                                                          "median": self.analysis.sifs_int["median"]["sif_k_ii"],
                                                          "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                              "sif_k_ii"]}
            json_dict['SIFs_integral']['T_interac'] = {"unit": "MPa",
                                                       "mean": self.analysis.sifs_int["mean"]["t_stress_int"],
                                                       "median": self.analysis.sifs_int["median"]["t_stress_int"],
                                                       "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                           "t_stress_int"]}
            json_dict['SIFs_integral']['K_I_Chen'] = {"unit": "MPa*m^{1/2}",
                                                      "mean": self.analysis.sifs_int["mean"]["k_i_chen"],
                                                      "median": self.analysis.sifs_int["median"]["k_i_chen"],
                                                      "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                          "k_i_chen"]}
            json_dict['SIFs_integral']['K_II_Chen'] = {"unit": "MPa*m^{1/2}",
                                                       "mean": self.analysis.sifs_int["mean"]["k_ii_chen"],
                                                       "median": self.analysis.sifs_int["median"]["k_ii_chen"],
                                                       "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                           "k_ii_chen"]}
            json_dict['SIFs_integral']['T_Chen'] = {"unit": "MPa",
                                                    "mean": self.analysis.sifs_int["mean"]["t_stress_chen"],
                                                    "median": self.analysis.sifs_int["median"]["t_stress_chen"],
                                                    "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                        "t_stress_chen"]}
            json_dict['SIFs_integral']['T_SDM'] = {"unit": "MPa",
                                                   "mean": self.analysis.sifs_int["mean"]["t_stress_sdm"],
                                                   "median": self.analysis.sifs_int["median"]["t_stress_sdm"],
                                                   "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                       "t_stress_sdm"]}
            json_dict['SIFs_integral']['J_I'] = {"unit": "N/mm",
                                                 "mean": self.analysis.sifs_int["mean"]["decomp_j_1"],
                                                 "median": self.analysis.sifs_int["median"]["decomp_j_1"],
                                                 "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                     "decomp_j_1"]}
            json_dict['SIFs_integral']['J_II'] = {"unit": "N/mm",
                                                  "mean": self.analysis.sifs_int["mean"]["decomp_j_2"],
                                                  "median": self.analysis.sifs_int["median"]["decomp_j_2"],
                                                  "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                      "decomp_j_2"]}
            json_dict['SIFs_integral']['J_III'] = {"unit": "N/mm",
                                                   "mean": self.analysis.sifs_int["mean"]["decomp_j_3"],
                                                   "median": self.analysis.sifs_int["median"]["decomp_j_3"],
                                                   "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                       "decomp_j_3"]}

            json_dict['SIFs_integral']['K_I_J'] = {"unit": "MPa*m^{1/2}",
                                                   "mean": self.analysis.sifs_int["mean"]["decomp_K_1"],
                                                   "median": self.analysis.sifs_int["median"]["decomp_K_1"],
                                                   "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"]["decomp_K_1"]}
            json_dict['SIFs_integral']['K_II_J'] = {"unit": "MPa*m^{1/2}",
                                                    "mean": self.analysis.sifs_int["mean"]["decomp_K_2"],
                                                    "median": self.analysis.sifs_int["median"]["decomp_K_2"],
                                                    "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                        "decomp_K_2"]}
            json_dict['SIFs_integral']['K_III_J'] = {"unit": "MPa*m^{1/2}",
                                                     "mean": self.analysis.sifs_int["mean"]["decomp_K_3"],
                                                     "median": self.analysis.sifs_int["median"]["decomp_K_3"],
                                                     "mean_wo_outliers": self.analysis.sifs_int["rej_out_mean"] [
                                                         "decomp_K_3"]}

            json_dict['Path_SIFs'] = {}
            json_dict['Path_SIFs']['J'] = {"unit": "N/mm",
                                           "result": list(np.asarray(self.analysis.path_results)[:, 0])}
            json_dict['Path_SIFs']['K_J'] = {"unit": "MPa*m^{1/2}",
                                             "result": list(np.asarray(self.analysis.path_results)[:, 1])}
            json_dict['Path_SIFs']['K_I'] = {"unit": "MPa*m^{1/2}",
                                             "result": list(np.asarray(self.analysis.path_results)[:, 2])}
            json_dict['Path_SIFs']['K_II'] = {"unit": "MPa*m^{1/2}",
                                              "result": list(np.asarray(self.analysis.path_results)[:, 3])}
            json_dict['Path_SIFs']['T_Chen'] = {"unit": "MPa",
                                                "result": list(np.asarray(self.analysis.path_results)[:, 4])}
            json_dict['Path_SIFs']['T_SDM'] = {"unit": "MPa",
                                               "result": list(np.asarray(self.analysis.path_results)[:, 5])}
            json_dict['Path_SIFs']['T_Int'] = {"unit": "MPa",
                                               "result": list(np.asarray(self.analysis.path_results)[:, 6])}
            json_dict['Path_SIFs']['J_1'] = {"unit": "N/mm",
                                             "result": list(np.asarray(self.analysis.path_results)[:, 7])}
            json_dict['Path_SIFs']['J_2'] = {"unit": "N/mm",
                                             "result": list(np.asarray(self.analysis.path_results)[:, 8])}
            json_dict['Path_SIFs']['J_3'] = {"unit": "N/mm",
                                             "result": list(np.asarray(self.analysis.path_results)[:, 9])}
            json_dict['Path_SIFs']['K_I_J'] = {"unit": "MPa*m^{1/2}",
                                               "result": list(np.asarray(self.analysis.path_results)[:, 10])}
            json_dict['Path_SIFs']['K_II_J'] = {"unit": "MPa*m^{1/2}",
                                                "result": list(np.asarray(self.analysis.path_results)[:, 11])}
            json_dict['Path_SIFs']['K_III_J'] = {"unit": "MPa*m^{1/2}",
                                                 "result": list(np.asarray(self.analysis.path_results)[:, 12])}

        if self.analysis.integral_properties.buckner_williams_terms is not None:
            json_dict['Bueckner_Chen_integral'] = {}
            terms = self.analysis.williams_int[0, :, 0]
            for i, term in enumerate(terms):
                json_dict['Bueckner_Chen_integral'][f'a_{term:.0f}'] = {"unit": unit_of_williams_coefficients(term),
                                                                        "mean": self.analysis.sifs_int["mean"] [
                                                                            "williams_int_a_n"][i],
                                                                        "median": self.analysis.sifs_int["median"] [
                                                                            "williams_int_a_n"][i],
                                                                        "mean_wo_outliers":
                                                                            self.analysis.sifs_int["rej_out_mean"] [
                                                                                "williams_int_a_n"][i]}
            for i, term in enumerate(terms):
                json_dict['Bueckner_Chen_integral'][f'b_{term:.0f}'] = {"unit": unit_of_williams_coefficients(term),
                                                                        "mean": self.analysis.sifs_int["mean"] [
                                                                            "williams_int_b_n"][i],
                                                                        "median": self.analysis.sifs_int["median"] [
                                                                            "williams_int_b_n"][i],
                                                                        "mean_wo_outliers":
                                                                            self.analysis.sifs_int["rej_out_mean"] [
                                                                                "williams_int_b_n"][i]}



            json_dict['Path_Williams_a_n'] = {}
            for i, term in enumerate(terms):
                json_dict['Path_Williams_a_n'][f'a_{term:.0f}'] = {"unit": unit_of_williams_coefficients(term),
                                                                   "result": list(self.analysis.williams_int_a_n[:, i])}
            json_dict['Path_Williams_b_n'] = {}
            for i, term in enumerate(terms):
                json_dict['Path_Williams_b_n'][f'b_{term:.0f}'] = {"unit": unit_of_williams_coefficients(term),
                                                                   "result": list(self.analysis.williams_int_b_n[:, i])}

        json_dict['Path_Properties'] = {}
        json_dict['Path_Properties']['NumOfNodes'] = {"unit": "1",
                                                      "result": list(self.analysis.num_of_path_nodes)}
        json_dict['Path_Properties']['TickSize'] = {"unit": "mm",
                                                    "result": list(self.analysis.tick_sizes)}
        json_dict['Path_Properties']['LineXL'] = {"unit": "mm",
                                                  "result": list(np.asarray(self.analysis.path_sizes)[:, 0])}
        json_dict['Path_Properties']['LineXR'] = {"unit": "mm",
                                                  "result": list(np.asarray(self.analysis.path_sizes)[:, 1])}
        json_dict['Path_Properties']['LineYB'] = {"unit": "mm",
                                                  "result": list(np.asarray(self.analysis.path_sizes)[:, 2])}
        json_dict['Path_Properties']['LineYT'] = {"unit": "mm",
                                                  "result": list(np.asarray(self.analysis.path_sizes)[:, 3])}

        top_offsets = []
        bot_offsets = []
        for each_path_index in range(len(self.analysis.path_results)):
            top_offsets.append(self.analysis.integration_points[each_path_index][1][-1])
            bot_offsets.append(self.analysis.integration_points[each_path_index][1][0])

        json_dict['Path_Properties']['TopOffset'] = {"unit": "mm",
                                                     "result": top_offsets}
        json_dict['Path_Properties']['BotOffset'] = {"unit": "mm",
                                                     "result": bot_offsets}

        json_dict['CrackPy_settings'] = {}
        sections = ['integral_properties', 'optimization_properties', 'crack_tip', 'material']
        objects = [self.analysis.integral_properties, self.analysis.optimization_properties, self.analysis.crack_tip,
                   self.analysis.material]

        for section, obj in zip(sections, objects):
            json_dict['CrackPy_settings'][section] = {}
            if obj is not None:
                for attr, value in vars(obj).items():
                    if not callable(value) and not attr.startswith('__'):
                        json_dict['CrackPy_settings'][section][attr] = value

        json_file = Path(self.json_path) / (Path(self.filename).stem + '.json')
        with open(json_file, 'w') as outfile:
            json.dump(json_dict, outfile, indent=4, default=str)

    @staticmethod
    def _make_path(output_path) -> Path:
        """Create and return path."""
        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _set_filename(self) -> str:
        """Transforms 'Filename.txt' -> 'Filename_right_Output.txt'"""
        return Path(self.analysis.nodemap_file).stem + '_' + self.analysis.crack_tip.left_or_right + '_Output.txt'
