import os
import shutil
import unittest
import tempfile

from crackpy.fracture_analysis.read import OutputReader


class TestOutputReader(unittest.TestCase):

    # Arrange
    def setUp(self):

        self.reader = OutputReader()
        self.path = os.path.join(  # '..', '..', '..', '..',
                                 'test_data', 'fracture_analysis', 'txt-files')

        self.files = os.listdir(self.path)
        self.possible_tags = ["CJP_results", "Williams_fit_results", "SIFs_integral", "Bueckner_Chen_integral",
                              "Path_SIFs", "Path_Williams_a_n", "Path_Williams_b_n", "Path_Properties"]

    # Act
    def test_read_tag_data(self):
        temp_dir = tempfile.mkdtemp()
        try:
            # iterate over all files in the folder
            for file in self.files:
                # iterate over all possible tags
                for tag in self.possible_tags:
                    df = self.reader.read_tag_data(path=self.path, filename=file, tag=tag)
                    df.to_csv(os.path.join(temp_dir, f'{file}_{tag}'))

        finally:
            shutil.rmtree(temp_dir)

    def test_make_csv_from_results(self):
        temp_dir = tempfile.mkdtemp()
        try:
            for file in self.files:
                for tag in self.possible_tags:
                    self.reader.read_tag_data(path=self.path, filename=file, tag=tag)

            # Export without filter
            self.reader.make_csv_from_results(files="all", output_path=temp_dir, output_filename='results.csv')

            # Export with filter
            self.reader.make_csv_from_results(files="all",
                                              output_path=temp_dir,
                                              output_filename="results_filtered_max_force.csv",
                                              filter_condition={"Force": (14900, 15100)})

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
