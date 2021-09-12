[![Documentation Status](https://readthedocs.org/projects/pytrack-ntu/badge/?version=latest)](https://pytrack-ntu.readthedocs.io/en/latest/?badge=latest)
<!-- [![Build Status](https://travis-ci.org/titoghose/PyTrack.svg?branch=master)](https://travis-ci.org/titoghose/PyTrack) -->
[![codecov](https://codecov.io/gh/titoghose/PyTrack/branch/master/graph/badge.svg)](https://codecov.io/gh/titoghose/PyTrack)

# PyTrack

This is a toolkit to analyse and visualize eye tracking data. It provides the following functionality:

## Feature Extraction
This involves extraction of parameters or meta-data related to blinks, fixations, saccades, microsaccades and pupil diameter. The features extracted are as follows:

| Blink        |  Fixations   | Saccades  | Microsaccades | Pupil            | Revisits to AOI/ROI  |
|------------  | -----------  | --------- | ------------- | ---------------- |--------------------- |
| Count        | Count        | Count     | Count         | Size             | Count                |
| Avg Duration | Avg Duration | Velocity  | Velocity      | Time to Peak     | First Pass Duration  |
| Max Duration | Max Duration | Amplitude | Amplitude     | Peak Size        | Second Pass Duration |
|              |              | Duration  | Duration      | Avg Size         |                      |
|              |              |           |               | Slope            |                      |
|              |              |           |               | Area Under Curve |                      |

## Statistical Analysis
After extraction of features, PyTrack can perform tests such as the student T-Test, Welch T-Test, ANOVA, RMANOVA, n-way ANOVA and Mixed ANOVA. The between and within group factors can be specified.

## Visualization
PyTrack can generate a variety of plots. The visualization is through an interactive GUI. The plots that can be generated are as follows:
1. Fixation plot
2. Individual subject gaze heat map
3. Aggregate subject gaze heat map
4. Dynamic pupil size and gaze plot
5. Microsaccade position and velocity plot
6. Microsaccade main sequence plot



# Table of Contents
- [PyTrack](#pytrack)
  - [Feature Extraction](#feature-extraction)
  - [Statistical Analysis](#statistical-analysis)
  - [Visualization](#visualization)
- [Table of Contents](#table-of-contents)
- [Documentation](#documentation)
- [Installation](#installation)
- [Sample Data](#sample-data)
- [Using PyTrack](#using-pytrack)
- [Advanced Functionality](#advanced-functionality)
  - [Statistical Tests](#statistical-tests)
  - [Accessing extracted features as a dictionary](#accessing-extracted-features-as-a-dictionary)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

# Documentation
The detailed documentation for the methods and classes of PyTrack can be found [here](https://pytrack-ntu.readthedocs.io/en/latest/)

# Installation

PyTrack is built for Python3 because support for the Python2 is going to be stopped at the end of 2019. In order to install PyTrack please use any of the following:

```
python3 -m pip install PyTrack-NTU
pip install PyTrack-NTU
pip3 install PyTrack-NTU
```

Please make sure that pip is for Python3 and not Python2. Python3  can be found [here](https://www.python.org/downloads/) or Anaconda Python3 can be found [here](https://www.anaconda.com/distribution/).

**NOTE:** Python3 can be installed alongside Python2

# Sample Data
In order to test the toolkit some sample data in SMI, EyeLink and Tobii formats can be found [here](https://osf.io/f9mey/files/). The .txt file in the folder describes the data found. The SMI and Tobii files have been taken from [here](http://www2.hu-berlin.de/eyetracking-eeg/testdata.html).

# Using PyTrack

The quickest and most concise way to get started is to go through the Python
Notebooks:
1. **getting_started_ExpMode.ipynb**
2. **getting_started_SAMode.ipynb**
3. **getting_started_OwnData.ipynb**: If you have data other than Tobii, SMI or EyeLink.

For some advanced use cases read on, and for viewing the detailed documentation of the
different modules see [here](https://pytrack-ntu.readthedocs.io/en/latest/).

# Advanced Functionality

## Statistical Tests
The Experiment class contains a function called analyse() which is used to perform statistical analysis (eg: ANOVA or T test), by default there is only 1 between group factor ("Subject_type") and 1 within group factor ("Stimuli_type") that is considered. If additional factors need to be considered they need to added to the json file.

* For example if Gender is to be considered as an additional between group factor then in the json file, under "Subjects", for each subject, a corresponding dicitionary must be created where you mention the factor name and the corresponding value. Please also note that the square brackets ('[', ']') after group type need to be changed to curly brackets ('{', '}').

* Similarly for Stimuli, for example, if you are showing Words and Pictures to elicit different responses from a user and you additonally have 2 different brightness levels ("High" and "Low") then mention Brightness as an additional within group factor.


```json
{
   "Subjects":{
      "group1":{
         "Subject_01": {"Gender": "M"},
         "Subject_02": {"Gender": "F"}
      },
      "group2":{
         "Subject_03": {"Gender": "F"},
         "Subject_04": {"Gender": "M"}
      }
   },
   "Stimuli":{
      "Type_1":{
         "Stim_1": {"Brightness": "High"},
         "Stim_2": {"Brightness": "Low"}
      },
      "Type_2":{
         "Stim_3": {"Brightness": "Low"},
         "Stim_4": {"Brightness": "High"}
      },
   },
}

```
Sample code segment to use the advanced statistical test:

```python
from PyTrack.Experiment import Experiment

exp = Experiment(json_file="abcd/efgh/NTU_Experiment/NTU_Experiment.json")

exp.metaMatrixInitialisation()

exp.analyse(parameter_list={"all"},
            between_factor_list=["Subject_type", "Gender"],
            within_factor_list=["Stimuli_type", "Brightness"],
            statistical_test="anova",
            file_creation=True)

```

## Accessing extracted features as a dictionary

In case you wish to get the extracted features for a particular Subject on a particular Stimulus:

```python
from PyTrack.Experiment import Experiment

exp = Experiment(json_file="complete/path/to/NTU_Experiment/NTU_Experiment.json")

subject_name = "sub_333" #specify your own subject's name (must be in json file)
stimulus_name = "Alpha1" #specify your own stimulus name (must be in json file)

# Access metadata dictionary for particular subject and stimulus
exp.metaMatrixInitialisation()
single_meta = exp.getMetaData(sub=subject_name,
                              stim=stimulus_name)

# Access metadata dictionary for particular subject and averaged for stimulus types
exp.metaMatrixInitialisation(average_flag=True)
agg_type_meta = exp.getMetaData(sub=subject_name,
                                 stim=None)

```

# Authors

* **Upamanyu Ghose** ([github](https://github.com/titoghose) | [email](titoghose@gmail.com))
* **Arvind A S** ([github](https://github.com/arvindas) | [email](96arvind@gmail.com))

See also the list of [contributors](https://github.com/titoghose/PyTrack/contributors) who participated in this project.

# License

This project is licensed under the GPL3 License - see the [LICENSE.txt](LICENSE.txt) file for details

# Acknowledgments

* We would like to thank [Dr. Dominique Makowski](https://dominiquemakowski.github.io/) for helping us develop this toolkit.

* The formatsBridge module was adapted from the work done by [Edwin Dalmaijer](https://github.com/esdalmaijer) in [PyGazeAnalyser](https://github.com/esdalmaijer/PyGazeAnalyser/).

* This work was done under the supervision of [Dr. Chng Eng Siong](http://www.ntu.edu.sg/home/aseschng/) - School of Computer Science and Engineering NTU and in collaboration with [Dr. Xu Hong](http://www.ntu.edu.sg/home/xuhong/) - School of Humanitites and Social Sciences NTU.

* We extend our thanks to the **Department of Computer Science and Engineering Manipal Isntitute of Technology**[[link]](https://manipal.edu/mit/department-faculty/department-list/computer-science-and-engineering.html) and the **Department of Computer Science and Information Systems BITS Pilani, Hyderabad Campus** [[link]](https://www.bits-pilani.ac.in/hyderabad/computerscience/ComputerScience).

