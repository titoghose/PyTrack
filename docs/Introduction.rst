Introduction to the framework
=============================

This is a framework to analyse and visualize eye tracking data. It
offers 2 designs of analysis:

-  **Experiment Design**: Analyse an entire experiment with 1 or more
   subjects/participants presented with 1 or more stimuli.
-  **Stand-alone Design**: Analyse a single stimulus for a single
   person.

As of now, it supports data collected using SR Research EyeLink, SMI and
Tobii eye trackers. The framework contains a *formatBridge* function
that converts these files into a base format and then performs analysis
on it.

Getting Started
---------------

These instructions will get you a copy of the project up and running on
your local machine for development and testing purposes.

Prerequisites
~~~~~~~~~~~~~

The framework uses Python 3.x. For systems running Linux or Mac OS, it
should be pre-installed. For Windows users, in case you do not have it
installed, get it from `here <https://www.python.org/downloads/>`_.

Installing
~~~~~~~~~~

To install **PyTrack** on your system, it can be done with one simple
step:

::

   pip install PyTrack

Running the tests
-----------------

In order to test **PyTrack**, some sample data files can be found
`here <https://drive.google.com/open?id=1N9ZrTO6Bikx3aI7BKivSFAp3vrLxSCM6>`__. To get started, first you need to choose which design you
want to run the framework in. If you wish to use the *Experiment
Design*, see `this <#experiment-design>`_. If you wish to use the *Stand-alone Design* see
`this <#stand-alone-design>`__.

Experiment Design
~~~~~~~~~~~~~~~~~

Setup
^^^^^

Before running the framework, lets setup the folder so **PyTrack** can
read and save all the generated figures in one central location and
things are organised.

Create a directory structure like the one shown below. It is essential
for the listed directories to be present for the proper functioning of
**PyTrack**.

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

*[Experiment-Name]* stands for the name of your experiment. Lets assume
that your experiment name is "*NTU_Experiment*". The rest of the steps
will use this alias as the *[Experiment-Name]* folder.

Now, follow these steps:

1. Place the data of all your subjects in the *Data* folder under the
   main *NTU_Experiment* folder. Make sure the name of each of the data
   files is the name of the subjects/paticipants. Replace all spaces( )
   with underscores (_).

   eg. *waffle_buttersnaps.asc* or *subject_001.asc*

2. For proper visualization of gaze data, its best if you include the
   stimuli presented during your experiment inside the *Stimuli* folder.
   Make sure the images have either **jpg, jpeg** or **png** extensions.

   eg. *stim_1.jpg* or *random_picture.png*

3. The last and final step to setup the experiment directory is to
   include the experiment description json file. This file should
   contain the essential details of your experiment. It contains
   specifications regarding your experiment suchas the stimuli you wish
   to analyse or the participants/subjects you wish to include.
   Mentioned below is the json file structure. The content below can be
   copied and pasted in a file called *NTU_Experiment*.json (basically
   the name of your experiment with a json extension).

   -  "*Experiment_name*" should be the same name as the json file
      without the extension and "*Path*" should be the absolute path to
      your experiment directory without the final "/" at the end.
   -  The subjects should be added under the "*Subjects*" field. You may
      specify one or more groups of division for your subjects
      (recommended for aggregate between group statistical analysis).
      **There must be atleast 1 group**.
   -  The stimuli names shpuld be added under the "*Stimuli*" field and
      again you may specify one or more types (recommended for aggregate
      between stimulus type statistical analysis). **There must be
      atleast 1 type**.
   -  The "*Control_Questions*" field is optional. In case you have some
      stimuli that should be used to standardise/normalise features
      extracted from all stimuli, sepcify the names here. **These
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
            "FixationSeq"   
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


Using PyTrack
^^^^^^^^^^^^^

This involves less than 10 lines of python code. However, in case you
want to do more detailed analysis, it may involve a few more lines.

Using *formatBridge* majorly has 3 cases.:

1. **Explicitly specify the stimulus order for each subject** as a list
   to the *generateCompatibleFormats* function. This case should be used
   when the order of stimuli is randomised for every participant. In
   this case, each participant needs a file specifying the stimulus
   presentation order. Hence, create a folder inside the *Data* folder
   called **stim** and place individual .txt files with the same names
   as the subject/participant names with the a new stimulus name on each
   line. Finally, the *stim_list_mode* parameter in the
   *generateCompatibleFormat* function needs to be set as "diff" (See
   `Example <#example-use>`_).

   eg. If subject data file is *subject_001.asc*, the file in the stim
   folder should be *subject_001.txt*

   *Note: Yes we undertsand this is a tedious task, but this is the only
   way we can understand the order of the stimulus which is needed for
   conclusive analysis and visualization.*\ **However, in case you are
   using EyeLink data, you can pass a message called "Stim Key:
   [stim_name]" during each stimulus and we can extract it
   automatically. See**\ *\ *\ `documentation <https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html>`_\ *\ *\ **.**

2. **Explicitly specify the stimulus order for the entire experiment**.
   This is for the case where the same order of stimuli are presented to
   all the participants. Just create a file called *stim_file.txt* and
   place it inside the *Data* folder. Finally, the *stim_list_mode*
   parameter in the *generateCompatibleFormat* function needs to be set
   as "common" (See `Example <#example-use>`_).

3. **Do not sepcify any stimulus order list**. In this case, the output
   of the statistical analysis will be inconclusive and the
   visualization of gaze will be on a black screen instead of the
   stimulus image. The *stim_list_mode* parameter in the
   *generateCompatibleFormat* function needs to be set as "NA". However,
   you can still extract the metadata and features extracted for each
   participant but the names will not make any sense. **WE DO NOT
   RECOMMEND THIS**.


Example Use
^^^^^^^^^^^

See `documentation <https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html>`_ for a detailed understanding of each function.

.. code:: python

   from PyTrack.formatBridge import generateCompatibleFormat
   from PyTrack.Experiment import Experiment

   # function to convert data to generate database in base format 
   generateCompatibleFormat(exp_path="abcd/efgh/NTU_Experiment/", device="eyelink", stim_list_mode='diff', start='start_trial', stop='stop_trial')


   # Creating an object of the Experiment class
   exp = Experiment(json_file="abcd/efgh/NTU_Experiment/NTU_Experiment.json")


   # Arvind has to add the function call for analysis
   ## CODE GOES HERE


   subject_name = "Sub_001"
   stimulus_name = "Stim_1"
   # Access metadata dictionary for particular subject and stimulus
   single_meta = exp.getMetaData(sub=subject_name, stim=stimulus_name)

   # Access metadata dictionary for particular subject and averaged for stimulus types
   agg_type_meta = exp.getMetaData(sub=subject_name, stim=None)


   # This function call opens up an interactive GUI that can be used to visualize the experiment data
   exp.visualizeData()


Stand-alone Design
~~~~~~~~~~~~~~~~~~~~~~
[In progress]


Authors
-------

-  **Upamanyu Ghose** (`github <https://github.com/titoghose>`_ \| `email <titoghose@gmail.com>`_)
-  **Arvind A S** (`github <https://github.com/arvindas>`__ \|
   `email <96arvind@gmail.com>`__)

See also the list of `contributors <https://github.com/titoghose/PyTrack/contributors>`_ who participated in this project.

License
-------

This project is licensed under the MIT License - see the `LICENSE.txt`
file for details

Acknowledgments
---------------

-  The formatsBridge module was adapted from the work done by `Edwin
   Dalmaijer <https://github.com/esdalmaijer>`_ in `PyGazeAnalyser <https://github.com/esdalmaijer/PyGazeAnalyser/>`_.

-  This work was done under the supervision of `Dr. Chng Eng Siong <http://www.ntu.edu.sg/home/aseschng/>`_ -
   School of Computer Science and Engineering NTU and in collaboration
   with `Dr. Xu Hong <http://www.ntu.edu.sg/home/xuhong/>`_ - School of Humanitites and Social Sciences NTU.

-  We extend our thanks to the **Department of Computer Science and
   Engineering Manipal Isntitute of Technology**\ `[link] <https://manipal.edu/mit/department-faculty/department-list/computer-science-and-engineering.html>`_ and the
   **Department of Computer Science and Information Systems BITS Pilani,
   Hyderabad Campus** `[link] <https://www.bits-pilani.ac.in/hyderabad/computerscience/ComputerScience>`__.
