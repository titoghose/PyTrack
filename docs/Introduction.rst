This is a toolkit to analyse and visualize eye tracking data. It
provides the following functionality:

Feature Extraction
------------------

This involves extraction of parameters or meta-data related to blinks,
fixations, saccades, microsaccades and pupil diameter. The features
extracted are as follows:

============ ============ ========= ============= ================ ====================
Blink        Fixations    Saccades  Microsaccades Pupil            Revisits to AOI/ROI
============ ============ ========= ============= ================ ====================
Count        Count        Count     Count         Size             Count
Avg Duration Avg Duration Velocity  Velocity      Time to Peak     First Pass Duration
Max Duration Max Duration Amplitude Amplitude     Peak Size        Second Pass Duration
\                         Duration  Duration      Avg Size
\                                                 Slope
\                                                 Area Under Curve
============ ============ ========= ============= ================ ====================

Statistical Analysis
--------------------

After extraction of features, PyTrack can perform tests such as the
student T-Test, Welch T-Test, ANOVA, RMANOVA, n-way ANOVA and Mixed
ANOVA. The between and within group factors can be specified.

Visualization
-------------

PyTrack can generate a variety of plots. The visualization is through an
interactive GUI. The plots that can be generated are as follows:

1. Fixation plot
2. Individual subject gaze heat map
3. Aggregate subject gaze heat map
4. Dynamic pupil size and gaze plot
5. Microsaccade position and velocity plot
6. Microsaccade main sequence plot

Table of Contents
=================

1. `Installation <#installation>`__
2. `Sample Data <#sample-data>`__
3. `Using PyTrack <#using-pytrack>`__

   1. `Setup <#setup>`__
   2. `Running PyTrack <#running-pytrack>`__

4. `Advanced Functionality <#advanced-functionality>`__

   1. `Statistical Tests <#statistical-tests>`__
   2. `Accessing extracted features as a
      dictionary <#accessing-extracted-features-as-a-dictionary>`__
   3. `Using PyTrack in Stand-alone
      mode <#using-pytrack-in-stand-alone-mode>`__

5. `Authors <#authors>`__
6. `Acknowledgments <#acknowledgments>`__


Installation
============

PyTrack is built for Python3 because support for the Python2 is going to
be stopped at the end of 2019. In order to install PyTrack please use
any of the following:

::

   python3 -m pip install PyTrack-NTU
   pip install PyTrack-NTU
   pip3 install PyTrack-NTU

Please make sure that pip is for Python3 and not Python2. Python3 can be
found `here <https://www.python.org/downloads/>`__ or Anaconda Python3
can be found `here <https://www.anaconda.com/distribution/>`__.

**NOTE:** Python3 can be installed alongside Python2

Sample Data
===========

In order to test the toolkit some sample data in SMI, EyeLink and Tobii
formats can be found
`here <https://osf.io/f9mey/files/>`__.
The .txt file in the folder describes the data found. The SMI and Tobii
files have been taken from
`here <http://www2.hu-berlin.de/eyetracking-eeg/testdata.html>`__.

Using PyTrack
=============

Setup
-----

Before running the framework, lets setup the folder so PyTrack can read
and save all the generated figures in one central location and things
are organised.

Create a directory structure like the one shown below. It is essential
for the listed directories to be present for the proper functioning of
PyTrack.

**NOTE:** The sample data has a folder called NTU_Experiment which is
already organised in the following manner. It can be used as reference.

::

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

*[Experiment-Name]* stands for the name of your experiment. The rest of
the steps will use *NTU_Experiment* as the *[Experiment-Name]* folder.

Now, follow these steps:

1. Place the data of all your subjects in the *Data* folder under the
   main *NTU_Experiment* folder. Make sure the name of each of the data
   files is the name of the subjects/paticipants. Replace all spaces( )
   with underscores (_).

   eg. *waffle_buttersnaps.asc* or *subject_001.asc*

2. For proper visualization of gaze data, its best if you include the
   stimuli presented during your experiment inside the *Stimuli* folder.
   Make sure the images have either **jpg, jpeg** or **png** extensions
   and the names match the names of the stimuli as present in your
   recorded data.

   eg. *stim_1.jpg* or *random_picture.png*

3. The last and final step to setup the experiment directory is to
   include the experiment description json file. This file should
   contain the essential details of your experiment. It contains
   specifications regarding your experiment such as the stimuli you wish
   to analyse or the participants/subjects you wish to include.
   Mentioned below is the json file structure. The content below can be
   copied and pasted in a file called *NTU_Experiment*.json

   -  "*Experiment_name*" should be the same name as the json file
      without the extension and "*Path*" should be the absolute path to
      your experiment directory without the final "/" at the end.
   -  The subjects should be added under the "*Subjects*" field. You may
      specify one or more groups of division for your subjects
      (recommended for between group statistical analysis). **There must
      be atleast 1 group**.
   -  The stimuli names should be added under the "*Stimuli*" field and
      again you may specify one or more types (recommended for
      between/within stimulus type statistical analysis). **There must
      be atleast 1 type**.
   -  The "*Control_Questions*" field is optional. In case you have some
      stimuli that should be used to standardise/normalise features
      extracted from all stimuli, specify the names here. **These
      stimuli must be present under the "Stimuli" field under one of the
      types**.
   -  **The field marked "Columns_of_interest" should not be altered**.
   -  Under "*Analysis_Params*", just change the values of
      "Sampling_Freq", "Display_height" and "Display_width" to match the
      values of your experiment.

   **Note**: If you wish to analyse only a subset of your stimuli or
   subjects, specify only the ones of interest in the json file. The
   analysis and visualization will be done only for the ones mentioned
   in the json file.

**NOTE:** A sample json file is present in the NTU_Experiment folder in
the sample data. You can just edit it to make your work simpler.

.. code:: json

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

Running PyTrack
---------------

**NOTE:** All sample segments shown below are for the NTU_Experiment
folder in the sample data.

1. In order to use the features, the **first step is to convert the raw
   data into a readable format**. In order to do so, the following code
   segment can be used:

   .. code:: python

      from PyTrack.formatBridge import generateCompatibleFormat

      generateCompatibleFormat(exp_path="complete/path/to/NTU_Experiment",
                              device="eyelink",
                              stim_list_mode='NA',
                              start='start_trial',
                              stop='stop_trial',
                              eye='B')

   To get a detailed understanding of the parameters of
   *generateCompatibleFormats* and modify it to your needs see the
   documentation
   `here <https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html#formatBridge.generateCompatibleFormat>`__.

2. The **second step is to create an object of the Experiment class**.

   .. code:: python

      from PyTrack.Experiment import Experiment

      # Creating an object of the Experiment class
      exp = Experiment(json_file="complete/path/to/NTU_Experiment/NTU_Experiment.json")

3. Now you can run the **feature extraction and statistical tests**

   .. code:: python

      # Instantiate the meta_matrix_dict of an Experiment to find and extract all features from the raw data
      exp.metaMatrixInitialisation()

      # Calling the function for the statistical analysis of the data
      exp.analyse(parameter_list={"all"},
                  between_factor_list=["Subject_type"],
                  within_factor_list=["Stimuli_type"],
                  statistical_test="anova",
                  file_creation=True)

      # Does not run any statistical test. Just saves all the data as csv files.
      exp.analyse(parameter_list={"all"},
                  statistical_test="None",
                  file_creation=True)

   To get a detailed understanding of the parameters of the *analyse*
   function:
   `here <https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html#experiment.analyse>`__

   To get a detailed understanding of the parameters of the
   *metaMatrixInitialisation* function:
   `here <https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html#experiment.metaMatrixInitialisation>`__

4. For **visualization**

   .. code:: python

      # This function call will open up a GUI which you can use to navigate the entire visualization process
      exp.visualizeData()

Advanced Functionality
======================

Statistical Tests
-----------------

The Experiment class contains a function called analyse() which is used
to perform statistical analysis (eg: ANOVA or T test), by default there
is only 1 between group factor ("Subject_type") and 1 within group
factor ("Stimuli_type") that is considered. If additional factors need
to be considered they need to added to the json file.

-  For example if Gender is to be considered as an additional between
   group factor then in the json file, under "Subjects", for each
   subject, a corresponding dicitionary must be created where you
   mention the factor name and the corresponding value. Please also note
   that the square brackets ('[', ']') after group type need to be
   changed to curly brackets ('{', '}').

-  Similarly for Stimuli, for example, if you are showing Words and
   Pictures to elicit different responses from a user and you
   additonally have 2 different brightness levels ("High" and "Low")
   then mention Brightness as an additional within group factor.

.. code:: json

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

Sample code segment to use the advanced statistical test:

.. code:: python

   from PyTrack.Experiment import Experiment

   exp = Experiment(json_file="abcd/efgh/NTU_Experiment/NTU_Experiment.json")

   exp.metaMatrixInitialisation()

   exp.analyse(parameter_list={"all"},
               between_factor_list=["Subject_type", "Gender"],
               within_factor_list=["Stimuli_type", "Brightness"],
               statistical_test="anova",
               file_creation=True)

Accessing extracted features as a dictionary
--------------------------------------------

In case you wish to get the extracted features for a particilar Subject
on a particular Stimulus:

.. code:: python

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

Using PyTrack in Stand-alone mode
---------------------------------

The stand-alone design requires only interaction with tyhe Stimulus
class. This is recommended if you wish to extract features or visualize
data for only 1 subject on a particular stimulus.

.. code:: python

   from PyTrack.Stimulus import Stimulus
   from PyTrack.formatBridge import generateCompatibleFormat
   import pandas as pd
   import numpy as np


   # function to convert data to generate csv file for data file recorded using EyeLink on both eyes and the stimulus name specified in the message section
   generateCompatibleFormat(exp_path="/path/to/smi_eyetracker_freeviewing.txt",
                           device="smi",
                           stim_list_mode='NA',
                           start='12',
                           stop='99')

   df = pd.read_csv("/path/to/smi_eyetracker_freeviewing.csv")

   # Dictionary containing details of recording. Please change the values according to your experiment. If no AOI is desired, set aoi value to [0, 0, Display_width, Display_height]
   sensor_dict = {
                     "EyeTracker":
                     {
                        "Sampling_Freq": 1000,
                        "Display_width": 1280,
                        "Display_height": 1024,
                        "aoi": [390, 497, 759, 732]
                     }
                  }

   # Creating Stimulus object. See the documentation for advanced parameters.
   stim = Stimulus(path="path/to/experiment/folder",
                  data=df,
                  sensor_names=sensor_dict)

   # Some functionality usage. See documentation of Stimulus class for advanced use.
   stim.findEyeMetaData()
   features = stim.sensors["EyeTracker"].metadata  # Getting dictioary of found metadata/features

   # Visualization of plots
   stim.gazePlot(save_fig=True)
   stim.gazeHeatMap(save_fig=True)
   stim.visualize()

   # Extracting features
   MS, ms_count, ms_duration = stim.findMicrosaccades(plot_ms=True)

See the stimulus class for more details on the functions:
`here <https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html#Stimulus.Stimulus>`__

Authors
=======

-  **Upamanyu Ghose** (`github <https://github.com/titoghose>`__ \|
   `email <titoghose@gmail.com>`__)
-  **Arvind A S** (`github <https://github.com/arvindas>`__ \|
   `email <96arvind@gmail.com>`__)

See also the list of
`contributors <https://github.com/titoghose/PyTrack/contributors>`__ who
participated in this project.


Acknowledgments
===============

-  The formatsBridge module was adapted from the work done by `Edwin
   Dalmaijer <https://github.com/esdalmaijer>`__ in
   `PyGazeAnalyser <https://github.com/esdalmaijer/PyGazeAnalyser/>`__.

-  This work was done under the supervision of `Dr. Chng Eng
   Siong <http://www.ntu.edu.sg/home/aseschng/>`__ - School of Computer
   Science and Engineering NTU and in collaboration with `Dr. Xu
   Hong <http://www.ntu.edu.sg/home/xuhong/>`__ - School of Humanitites
   and Social Sciences NTU.

-  We extend our thanks to the **Department of Computer Science and
   Engineering Manipal Isntitute of
   Technology**\ `[link] <https://manipal.edu/mit/department-faculty/department-list/computer-science-and-engineering.html>`__
   and the **Department of Computer Science and Information Systems BITS
   Pilani, Hyderabad Campus**
   `[link] <https://www.bits-pilani.ac.in/hyderabad/computerscience/ComputerScience>`__.

