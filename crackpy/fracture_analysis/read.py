import os.path
import pandas as pd
import numpy as np


def is_stringfloat(element: str) -> bool:
    """Checks if element can be converted to a float"""
    try:
        float(element)
        return True
    except ValueError:
        return False


class OutputReader:
    """This class is designed to read tagged data from fracture analysis output file **filename** in folder **path**.

    Methods:
        * read_tag_data - data is read into a pandas Dataframe
        * make_csv_from_results - saves data in csv file

    """

    def __init__(self):
        self.path = None
        self.filename = None
        self.possible_tags = None
        self.data = {}

    def read_tag_data(self, path: str or os.PathLike, filename: str, tag: str) -> pd.DataFrame:
        """Read data into Pandas dataframe and saves results to results dictionary

        Args:
            path: path to the output file
            filename: output file's name
            tag: tag which should be filtered

        Returns:
            df: dataframe with columns and values

        """
        if tag not in self._search_for_tags(filename=filename, path=path):
            raise ValueError(f"The tag {tag} does not exist! \n"
                             f"Possible tags: {self.possible_tags}")

        with open(os.path.join(path, filename), 'r') as text_file:
            read_header = False
            read_values = False

            try:
                _ = self.data[filename]
            except KeyError:
                self.data.update({filename: {}})

            for line in text_file:
                if '</' + tag + '>' in line:
                    break
                if '<' + tag + '>' in line:
                    read_header = True
                    continue

                if read_header:
                    # read header of tagged content
                    columns = line.strip('\n').strip(' ').split(',')
                    columns = [element.strip(' ') for element in columns]
                    df = pd.DataFrame(columns=columns)
                    read_header = False
                    read_values = True
                    continue
                if read_values:
                    # convert to float if possible
                    values = []
                    for val in line.strip('\n').split(','):
                        val = val.strip(' ')
                        if is_stringfloat(val):
                            val = float(val)
                        values.append([val])

                    # read values of tagged content
                    columns_to_values = pd.DataFrame.from_dict(dict(zip(columns, values)))
                    df = pd.concat([df, columns_to_values], ignore_index=True)

                    # save to results
                    self.data[filename].update({tag: df})

                # always read meta data
            if "Experiment_data" not in self.data[filename].keys():
                _ = self.read_tag_data(path, filename, "Experiment_data")
            return df

    def make_csv_from_results(self, files: list or str, output_path: str or os.PathLike, output_filename: str,
                              tags: list or str = "all", filter_condition: dict or None = None):
        """
        Writes data for a list of files to a csv output.

        Args:
            files: list of files to be exported. If 'all', each calculated nodemap is exported
            output_path: path to folder where result data frame should be stored (has to exist!)
            output_filename: filename
            tags: list of tags to be exported. Exports all calculated tag results if 'all'
            filter_condition: dict = {DataType: (min_value, max_value)]
                              Filter Condition for csv export. Exports ONLY stages which fulfill the condition.
                              E.g. {'Force': (14500.0, 15500.0)}

        """
        # sanity check for filter_condition
        if filter_condition is not None:
            for key in filter_condition:
                if not isinstance(key, str):
                    raise TypeError("The filter_condition key 'Data type' should be a string.")
                if not isinstance(filter_condition[key], tuple):
                    raise TypeError("The filter_condition value should be a tuple of flaots.")
                if not isinstance(filter_condition[key][0], int or float) \
                        or not isinstance(filter_condition[key][1], int or float):
                    raise TypeError("The filter_condition values should be a tuple of two floats.")
                if filter_condition[key][0] >= filter_condition[key][1]:
                    raise ValueError("The first entry of values for the filter condition should be the minimum value,\n"
                                     f"the second should be the maximum value. But they are {filter_condition[key]}")

        # check if results dictionary is empty
        if self.data == {} or None:
            raise ValueError("Results dictionary is empty. Use 'read_tag_data(path, file, tag) to fill the dictionary.")
        if files == "all":
            files = list(self.data.keys())

        # check which tags were read
        read_tags = list(self.data[files[0]].keys())
        if tags == "all":
            tags = read_tags
        elif any(tags) not in read_tags:
            raise ValueError("One of the tags you want to export to csv was not read before. Add the tag in the "
                             f"read_tag_data method. So far you read {read_tags}.")

        filenames = []
        all_results = []
        for filename in sorted(files):
            # get experiment data from dictionary or from file
            experiment_data = self.data[filename]["Experiment_data"]

            # check if filter condition meets
            if filter_condition is not None:
                for key in filter_condition:
                    if key not in experiment_data["Param"].to_list():
                        raise ValueError(f"The key {key} of your filter conditions for exporting data "
                                         f"to csv is not a valid filter condition.\n"
                                         f"Should be one of {experiment_data['Param'].to_list()}")

            if filter_condition is not None and not self._filtered_by_condition(filter_condition,
                                                                                experiment_data=experiment_data):
                continue

            # if filter condition holds we restructure data and export to csv
            filenames.append(filename)
            stage_params = []
            stage_results = []

            # export experiment data first
            stage_params.extend(self._restructure_results(experiment_data)[0])
            stage_results.extend(self._restructure_results(experiment_data)[1])

            # then export calculated tags
            for tag in tags:
                if tag == "Experiment_data":
                    pass
                else:
                    df = self.data[filename][tag]
                    # case check for different data structures
                    if "integral" in tag:
                        stage_params.extend(self._restructure_integral_df(df, tag)[0])
                        stage_results.extend(self._restructure_integral_df(df, tag)[1])
                    elif "Path" in tag:
                        stage_params.extend(self._restructure_path_statistics(df, tag)[0])
                        stage_results.extend(self._restructure_path_statistics(df, tag)[1])
                    else:
                        stage_params.extend(self._restructure_results(df, tag)[0])
                        stage_results.extend(self._restructure_results(df, tag)[1])

            # export the dataframe to csv
            all_results.append(stage_results)
        try:
            np_all_results = np.asarray(all_results)
            all_results_df = pd.DataFrame(np_all_results, columns=stage_params, index=filenames)
            all_results_df.to_csv(os.path.join(output_path, output_filename), index_label="filename")
        except UnboundLocalError:
            print(f"Filter condition is not satisfied by any file in files {files}.")

    @staticmethod
    def _filtered_by_condition(filter_condition: dict, experiment_data: pd.DataFrame) -> bool:
        """

        Args:
            filter_condition: the filter condition which is checked [Datatype, Value, Tolerance]
            experiment_data: dataframe which is checked on the condition

        Returns: True if condition holds, False else

        """
        pars = experiment_data["Param"].to_list()
        vals = experiment_data["Result"].to_list()
        filter_check_list = []
        for filter_type, condition in filter_condition.items():
            val = [vals[i] for i in range(len(pars)) if pars[i] == filter_type][0]
            if condition[0] <= val <= condition[1]:
                filter_check_list.append(True)
            else:
                filter_check_list.append(False)
        if all(filter_check_list):
            return True
        else:
            return False

    @staticmethod
    def _restructure_integral_df(df: pd.DataFrame, tag: str or None) -> tuple:
        """
        Internal method to restructure the Dataframe for integral output data.

        Args:
            df: obj of class DataFrame
            tag: tag

        Returns:
            a list for all parameter names for given tag and all values for these parameters

        """
        results = []
        params = []
        param_keys = df["Param"].to_list()
        means = df["Mean"].to_list()
        medians = df["Median"].to_list()
        means_wo_outliers = df["Mean_wo_outliers"].to_list()
        for param_index in range(len(param_keys)):
            if tag is not None:
                param = tag + "_" + param_keys[param_index]
            else:
                param = param_keys[param_index]
            params.append(param + "_mean")
            params.append(param + "_median")
            params.append(param + "_mean_wo_outliers")
            results.append(means[param_index])
            results.append(medians[param_index])
            results.append(means_wo_outliers[param_index])
        return params, results

    @staticmethod
    def _restructure_path_statistics(df: pd.DataFrame, tag: str) -> tuple:
        """Internal method to restructure the Dataframe for path dependent output data.

        Args:
            df: obj of class DataFrame
            tag: tag

        Returns:
            a list for all parameter names for given tag and all values for these parameters

        """
        params = []
        results = []
        for col in df.columns:

            # get col name
            name = col.split(' ')[0]

            # calculate path statisitics
            stat_attributes = {
                'mean': np.mean(df[col]),
                'median': np.median(df[col]),
                'quantile10': np.quantile(df[col], .10),
                'quantile90': np.quantile(df[col], .90),

                'max': np.max(df[col]),
                'min': np.min(df[col])
            }

            for param in stat_attributes.keys():
                params.append(f"{tag}_{name}_{param}")
                results.append(stat_attributes[param])
        return params, results

    @staticmethod
    def _restructure_results(df: pd.DataFrame, tag: str = None) -> tuple:
        """Internal method to restructure the Dataframe for path independent output data.

        Args:
            df: obj of class DataFrame
            tag: tag

        Returns:
            a list for all parameter names for given tag and all values for these parameters

        """
        params = df["Param"].to_list()
        results = df["Result"].to_list()
        for param_index in range(len(params)):
            if tag is not None:
                params[param_index] = tag + "_" + params[param_index]
        return params, results

    def _search_for_tags(self, filename: str, path: str or os.PathLike) -> list:
        """
        Internal Method to search for any possible tag in a given filename and path.

        Args:
            filename: filename of output file
            path: path to this file

        Returns:
            list of possible tags

        """
        if self.possible_tags is None:
            tag_list = []
            with open(os.path.join(path, filename)) as file:
                for line in file:
                    if '<' in line and '>' in line and '/' not in line:
                        tag = line.strip('<>\n')
                        tag_list.append(tag)
            self.possible_tags = tag_list
        else:
            pass
        return self.possible_tags
