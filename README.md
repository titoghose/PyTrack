[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f57df86d1eb94be0b150f45e16977566)](https://app.codacy.com/app/titoghose/PyTrack?utm_source=github.com&utm_medium=referral&utm_content=titoghose/PyTrack&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/titoghose/PyTrack.svg?branch=master)](https://travis-ci.org/titoghose/PyTrack)
[![codecov](https://codecov.io/gh/titoghose/PyTrack/branch/master/graph/badge.svg)](https://codecov.io/gh/titoghose/PyTrack)
[![Documentation Status](https://readthedocs.org/projects/pytrack-ntu/badge/)](https://pytrack-ntu.readthedocs.io/en/latest/)

# PyTrack

# Table of Contents
1. [Documentation](#documentation)
2. [Getting Started](#getting-started)
   1. [Prerequisites](#prerequisites)
   2. [Installing](#installing)
3. [Running the tests](#running-the-tests)
   1. [Experiment Design](#experiment-design)
      1. [Setup](#setup)
      2. [Using PyTrack](#using-pytrack)
      3. [Example Use](#example-use)
      4. [Advanced Functionality](#advanced-functionality)
   2. [Stand-alone Design](#stand-alone-design)
4. [Authors](#authors)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)


This is a framework to analyse and visualize eye tracking data. It offers 2 designs of analysis:
* **Experiment Design**: Analyse an entire experiment with 1 or more subjects/participants presented with 1 or more stimuli.
* **Stand-alone Design**: Analyse a single stimulus for a single person.


As of now, it supports data collected using SR Research EyeLink, SMI and Tobii eye trackers. The framework contains a *formatBridge* function that converts these files into a base format and then performs analysis on it.


## Documentation

The detailed documentation of the project with explanation of all its modules can be found [here](https://pytrack-ntu.readthedocs.io/en/latest/).


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The framework uses Python 3.x. For systems running Linux or Mac OS, it should be pre-installed. For Windows users, in case you do not have it installed, get it from [here](https://www.python.org/downloads/).


### Installing

To install ***PyTrack*** on your system, it can be done with one simple step:

```
pip install PyTrack-NTU
```

PyTrack uses Python 3.x. Hence, use pip for Python 3.x.

## Running the tests

In order to test ***PyTrack***, some sample data files can be found [here](https://drive.google.com/open?id=1tWD69hurELVuVRFzizCbukWnr22RZrnp).

To get started, first you need to choose which design you want to run the framework in. If you wish to use the *Experiment Design*, see [this](#experiment-design). If you wish to use the *Stand-alone Design* see [this](#stand-alone-design).

### Experiment Design

#### Setup

Before running the framework, lets setup the folder so ***PyTrack*** can read and save all the generated figures in one central location and things are organised.

Create a directory structure like the one shown below. It is essential for the listed directories to be present for the proper functioning of ***PyTrack***.
```
[Experiment-Name]
|
└── Data/
│   │   subject_001.[asc/txt/tsv/...]
│   │   subject_002.[asc/txt/tsv/...]
|   |__ ......
│
└── Stimulus/
│   │   stim_1.[jpg/jpeg]
│   │   stim_2.[jpg/jpeg]
|   |__ ......
|
└── [Experiment-Name].json

```
*[Experiment-Name]* stands for the name of your experiment. Lets assume that your experiment name is "*NTU_Experiment*". The rest of the steps will use this alias as the *[Experiment-Name]* folder.

Now, follow these steps:

1. Place the data of all your subjects in the *Data* folder under the main *NTU_Experiment* folder. Make sure the name of each of the data files is the name of the subjects/paticipants. Replace all spaces( ) with underscores (_).

    eg. *waffle_buttersnaps.asc* or *subject_001.asc*

2. For proper visualization of gaze data, its best if you include the stimuli presented during your experiment inside the *Stimuli* folder. Make sure the images have either **jpg, jpeg** or **png** extensions.

    eg. *stim_1.jpg* or *random_picture.png*

3. The last and final step to setup the experiment directory is to include the experiment description json file. This file should contain the essential details of your experiment. It contains specifications regarding your experiment suchas the stimuli you wish to analyse or the participants/subjects you wish to include. Mentioned below is the json file structure. The content below can be copied and pasted in a file called *NTU_Experiment*.json (basically the name of your experiment with a json extension).

    * "*Experiment_name*" should be the same name as the json file without the extension and "*Path*" should be the absolute path to your experiment directory without the final "/" at the end.
    * The subjects should be added under the "*Subjects*" field. You may specify one or more groups of division for your subjects (recommended for aggregate between group statistical analysis). **There must be atleast 1 group**.
    * The stimuli names should be added under the "*Stimuli*" field and again you may specify one or more types (recommended for aggregate between stimulus type statistical analysis). **There must be atleast 1 type**.
    * The "*Control_Questions*" field is optional. In case you have some stimuli that should be used to standardise/normalise features extracted from all stimuli, sepcify the names here. **These stimuli must be present under the "*Stimuli*" field under one of the types**.
    * **The field marked "*Columns_of_interest*" should not be altered**.
    * Under "*Analysis_Params*", just change the values of "Sampling_Freq", "Display_height" and "Display_width" to match the values of your experiment.

    ***Note***: If you wish to analyse only a subset of your stimuli or subjects, specify only the ones of interest in the json file. The analysis and visualization will be done only for the ones mentioned in the json file.

```json
{
   "Experiment_name":"NTU_Experiment",
   "Path":"abcd/efgh/NTU_Experiment",
   "Subjects":{
      "group1":[
         "Subject_01",
         "Subject_02"
      ],
      "group2":[
         "Subject_03",
         "Subject_04"
      ]
   },
   "Stimuli":{
      "Type_1":[
         "Stim_1",
         "Stim_2"
      ],
      "Type_2":[
         "Stim_3",
         "Stim_4"
      ],
   },
   "Control_Questions":[
         "Stim_1"
    ],
   "Columns_of_interest":{
      "EyeTracker":[
         "GazeLeftx",
         "GazeLefty",
         "GazeRightx",
         "GazeRighty",
         "PupilLeft",
         "PupilRight",
         "FixationSeq",
         "GazeAOI"
      ],
      "Extra":[
         "EventSource"
      ]
   },
   "Analysis_Params":{
      "EyeTracker":{
        "Sampling_Freq": 1000,
        "Display_width": 1920,
        "Display_height": 1280
      }
   }
}

```
**NOTE: For some advanced functionality on analysis read [ADVANCED FUNCTIONALITY](#advanced-functionality). If only basic functionality is desired, you may ignore it.**


#### Using PyTrack

This involves less than 10 lines of python code. However, in case you want to do more detailed analysis, it may involve a few more lines.

Using *formatBridge* majorly has 3 cases.:

1. **Explicitly specify the stimulus order for each subject** as a list to the *generateCompatibleFormats* function. This case should be used when the order of stimuli is randomised for every participant. In this case, each participant needs a file specifying the stimulus presentation order. Hence, create a folder inside the *Data* folder called ***stim*** and place individual .txt files with the same names as the subject/participant names with the a new stimulus name on each line. Finally, the *stim_list_mode* parameter in the *generateCompatibleFormat* function needs to be set as "diff" (See [Example](#example-use)).

   eg. If subject data file is *subject_001.asc*, the file in the stim folder should be *subject_001.txt*

   *Note: Yes we understand this is a tedious task, but this is the only way we can understand the order of the stimulus which is needed for conclusive analysis and visualization. **However, if you specify the stimulus name for every event in the message column of your data in this format: "Stim Key: [stim_name]", we can extract it automatically. WE RECOMMEND THIS FOR BEST USER EXPERIENCE.***

2. **Explicitly specify the stimulus order for the entire experiment**. This is for the case where the same order of stimuli are presented to all the participants. Just create a file called *stim_file.txt* and place it inside the *Data* folder. Finally, the *stim_list_mode* parameter in the *generateCompatibleFormat* function needs to be set as "common" (See [Example](#example-use)).

3. **Do not sepcify any stimulus order list**. In this case, the output of the statistical analysis will be inconclusive and the visualization of gaze will be on a black screen instead of the stimulus image. The *stim_list_mode* parameter in the *generateCompatibleFormat* function needs to be set as "NA". However, you can still extract the metadata and features extracted for each participant but the names will not make any sense. ***WE DO NOT RECOMMEND THIS***.


#### Example Use

See [documentation](https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html) for a detailed understanding of each function.

**Converting to the correct format:**

```python
from PyTrack.formatBridge import generateCompatibleFormat

# function to convert data to generate database in base format for experiment done using EyeLink on both eyes and the stimulus name specified in the message section
generateCompatibleFormat(exp_path="abcd/efgh/NTU_Experiment/",
                        device="eyelink",
                        stim_list_mode='NA',
                        start='start_trial',
                        stop='stop_trial',
                        eye='B')

```

**Running the analysis or extracting data:**

```python
from PyTrack.Experiment import Experiment

# Creating an object of the Experiment class
exp = Experiment(json_file="abcd/efgh/NTU_Experiment/NTU_Experiment.json")

# Instantiate the meta_matrix_dict of an Experiment to find and extract all features from the raw data
exp.metaMatrixInitialisation(standardise_flag=False,
                              average_flag=False)

# Calling the function for the statistical analysis of the data
# file_creation=True. Hence, the output of the data used to run the tests and the output of the tests will be stored in in the 'Results' folder inside your experiment folder
exp.analyse(parameter_list={"all"},
            between_factor_list=["Subject_type"],
            within_factor_list=["Stimuli_type"],
            statistical_test="anova",
            file_creation=True)

# Does not run any test. Just saves all the data as csv files.
exp.analyse(parameter_list={"all"},
            statistical_test="None",
            file_creation=True)

```

**Visualizing the data:**

```python
from PyTrack.Experiment import Experiment

# Creating an object of the Experiment class
exp = Experiment(json_file="abcd/efgh/NTU_Experiment/NTU_Experiment.json")

# This function call will open up a GUI which you can use to navigate the entire visualization process
exp.visualizeData()

```


#### Advanced Functionality

**THIS SECTION IS ONLY FOR ADVANCED STATISTICAL ANALYSIS FUNCTIONALITY. IGNORE IT IF THE BASIC ANALYSIS IS SUFFICIENT FOR YOU.**

The Experiment class contains a function called analyse() which is used to perform statistical analysis (eg: ANOVA or T test), by default there is only 1 between group factor ("Subject_type") and 1 within group factor ("Stimuli_type") that is considered. If additional factors need to be considered they need to added to the json file.

* For example if Gender is to be considered as an additional between group factor then in the json file, under "Subjects", for each subject, a corresponding dicitionary must be created where you mention the factor name and the corresponding value (eg: Subject_name: {"Gender" : "M"}). Please also note that the square brackets ('[', ']') after group type need to be changed to curly brackets ('{', '}').
* This must be similarly done for Stimuli, if any additional within group factor that describes the stimuli needs to be added. For example, if you are showing WORDS and PICTURES to elicit different responses from a user and you additonally have 2 different brightness levels ("High" and "Low") of the stimuli, you could consider Type1 and Type2 to be the PICTuRE and WORD gropus and mention Brightness as an additional within group factor.

The below code snippet just shows the changes that are to be done for Subject and Stimuli sections of the json file, the other sections remain the same.

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

**The snippet at the bottom allows the use of advanced functionality:**

```python
from PyTrack.Experiment import Experiment

# Creating an object of the Experiment class
exp = Experiment(json_file="abcd/efgh/NTU_Experiment/NTU_Experiment.json")

# Instantiate the meta_matrix_dict of an Experiment to find and extract all features from the raw data
exp.metaMatrixInitialisation(standardise_flag=False,
                              average_flag=False)

# Calling the function for advanced statistical analysis of the data
# file_creation=True. Hence, the output of the data used to run the tests and the output of the tests will be stored in in the 'Results' folder inside your experiment folder

#############################################################
## 1. Running anova on advanced between and within factors ##
#############################################################
exp.analyse(parameter_list={"all"},
            between_factor_list=["Subject_type", "Gender"],
            within_factor_list=["Stimuli_type", "Brightness"],
            statistical_test="anova",
            file_creation=True)

#############################################################
## 2. Running no tests. Just storing analysis data in Results folder ##
#############################################################
exp.analyse(statistical_test="None",
            file_creation=True)


# In case you want the data for a particular participant/subject as a dictionary of values, use this

subject_name = "Sub_001" #specify your own subject's name (must be in json file)
stimulus_name = "Stim_1" #specify your own stimulus name (must be in json file)

# Access metadata dictionary for particular subject and stimulus
single_meta = exp.getMetaData(sub=subject_name,
                              stim=stimulus_name)

# Access metadata dictionary for particular subject and averaged for stimulus types
agg_type_meta = exp.getMetaData(sub=subject_name,
                                 stim=None)

```


### Stand-alone Design

The stand-alone design requires only interaction with tyhe Stimulus class. This is recommended if you wish to extract features or visualize data for only 1 subject on a particular stimulus. If not, look at [Experiment Design](#experiment-design)

**Here is a sample code snippet explaining the functionality:**

```python
from PyTrack.Stimulus import Stimulus
from PyTrack.formatBridge import generateCompatibleFormat
import pandas as pd
import numpy as np


# function to convert data to generate csv file for data file recorded using EyeLink on both eyes and the stimulus name specified in the message section
generateCompatibleFormat(exp_path="/path/to/data/file/in/raw/format",
                        device="eyelink",
                        stim_list_mode='NA',
                        start='start_trial',
                        stop='stop_trial',
                        eye='B')

df = pd.read_csv("/path/to/enerated/data/file/in/csv/format")

# Dictionary containing details of recording. Please change the values according to your experiment. If no AOI is desired, set aoi_left values to (0, 0) and aoi_right to the same as Display_width and Display_height
sensor_dict = {
                  "EyeTracker":
                  {
                     "Sampling_Freq": 1000,
                     "Display_width": 1280,
                     "Display_height": 1024,
                     "aoi_left_x": 390,
                     "aoi_left_y": 497,
                     "aoi_right_x": 759,
                     "aoi_right_y": 732
                  }
               }

# Creating Stimulus object. See the documentation for advanced parameters.
stim = Stimulus(path="path/to/experiment/folder",
               data=df,
               sensor_names=sensor_dict)

# Some functionality usage. See documentation of Stimulus class for advanced use.
stim.findEyeMetaData()
features = stim.sensors["EyeTracker"].metadata  # Getting dictioary of found metadata/features

stim.gazePlot(save_fig=True)
stim.gazeHeatMap(save_fig=True)
stim.findMicrosaccades(plot_ms=True)
stim.visualize()

```

## Authors

* **Upamanyu Ghose** ([github](https://github.com/titoghose) | [email](titoghose@gmail.com))
* **Arvind A S** ([github](https://github.com/arvindas) | [email](96arvind@gmail.com))

See also the list of [contributors](https://github.com/titoghose/PyTrack/contributors) who participated in this project.

## License

This project is licensed under the GNU GPL v3 License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

* The formatsBridge module was adapted from the work done by [Edwin Dalmaijer](https://github.com/esdalmaijer) in [PyGazeAnalyser](https://github.com/esdalmaijer/PyGazeAnalyser/).

* This work was done under the supervision of [Dr. Chng Eng Siong](http://www.ntu.edu.sg/home/aseschng/) - School of Computer Science and Engineering NTU and in collaboration with [Dr. Xu Hong](http://www.ntu.edu.sg/home/xuhong/) - School of Humanitites and Social Sciences NTU.
* We extend our thanks to the **Department of Computer Science and Engineering Manipal Isntitute of Technology**[[link]](https://manipal.edu/mit/department-faculty/department-list/computer-science-and-engineering.html) and the **Department of Computer Science and Information Systems BITS Pilani, Hyderabad Campus** [[link]](https://www.bits-pilani.ac.in/hyderabad/computerscience/ComputerScience).
