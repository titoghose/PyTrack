Getting started
=================

- `Installation <#installation>`__
- `Sample Data <#sample-data>`__
- `Using PyTrack <#using-pytrack>`__
   
   - `Setup <#setup>`__
   - `Running PyTrack <#running-pytrack>`__

- `Advanced Functionality <#advanced-functionality>`__
   
   - `Statistical Tests <#statistical-tests>`__
   - `Accessing extracted features as a dictionary <#accessing-extracted-features-as-a-dictionary>`__
   - `Using PyTrack in Stand-alone mode <#using-pytrack-in-stand-alone-mode>`__

- `Authors <#authors>`__
- `Acknowledgments <#acknowledgments>`__


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

**NOTE:** Python3 can be installed alongside Python2. The best option is to use
pyenv.

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

The quickest and most concise way to get started is to go through the `Python
Notebooks <https://github.com/titoghose/PyTrack>`__:

-  **getting_started_ExpMode.ipynb**
-  **getting_started_SAMode.ipynb**
-  **getting_started_OwnData.ipynb**: If you have data other than Tobii, SMI or EyeLink.

For some advanced use cases read on, and for viewing the detailed documentation
see the `Modules <https://pytrack-ntu.readthedocs.io/en/latest/PyTrack.html#modules>`__.

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

