# PyTrack

This is a framework to analyse and visualize eye tracking data. It offers 2 designs of analysis:
* **Experiment Design**: Analyse an entire experiment with 1 or more subjects/participants presented with 1 or more stimuli.
* **Stand-alone Design**: Analyse a single stimulus for a single person. 

As of now, it supports data collected using SR Research EyeLink, SMI and Tobii eye trackers. The framework contains a *formatBridge* function that converts these files into a base format and then performs analysis on it. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The framework uses Python 3.x. For systems running Linux or Mac OS, it should be pre-installed. For Windows users, in case you do not have it installed, get it from [here](https://www.python.org/downloads/).


### Installing

To install ***PyTrack*** on your system, it can be done with one simple step:

```
pip install PyTrack
```

## Running the tests

In order to test ***PyTrack***, some sample data files can be found [here](). To get started, first you need to choose which design you want to run the framework in. If you wish to use the *Experiment Design*, see [this](#experiment-design). If you wish to use the *Stand-alone Design* see [this](#stand-alone-design).

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
    * The stimuli names shpuld be added under the "*Stimuli*" field and again you may specify one or more types (recommended for aggregate between stimulus type statistical analysis). **There must be atleast 1 type**.
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

```

#### Using ***PyTrack***

```python
from PyTrack.formatBridge import generateCompatibleFormat
from PyTrack.Experiment import Experiment

# function to convert data to generate database in base format 
generateCompatibleFormat(exp_path="abcd/efgh/NTU_Experiment/", device="eyelink)


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
```

### Stand-alone Design

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
