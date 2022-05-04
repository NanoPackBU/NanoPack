# NanoView_G33
NanoPack, the NanoView senior design project made by Group 33.  We are providing a packaging machine which moves chips from one container to another so that this process can be performed faster and with fewer errors.  Powering this is a CNC-style pick and place machine equipped with two cameras that allow us to leverage Computer Vision and Machine Learning models to determine the location of the objects and the numbers on the chips.

## Current Status of the Project
Our team is currently working on tuning and testing the machine with the tweezer-supporting linear actuator. This actuator replaced the vacuum pump that was used during the final prototype testing. In order to use the tweezers, we had to change the orientation of the clamshell contraints, which required us to retrain our YOLO model for object detection. We are currently in the process of testing the new machine learning model and fine-tuning the new configuration distances. This tuning requires more precision than what was required for the vacuum pump due to the small distances between chips, as well as the presence of a low-hanging solenoid responsible for providing the force needed to close the tweezers. If we find that the tweezers are unable to consistantly pick up chips, we have the ability to quickly go back to using the vacuum pump. We are also in the process of transferring all of our material and documentation to NanoView, who will be receiving our machine the week after ECE Day on May 6th.

## Installation Information
Information on how to install our repository and use its software is found on our github wiki on the page [Installation, setup, and support](https://github.com/BostonUniversitySeniorDesign/NanoView_G33/wiki/Installation-Setup-and-Support). 

## File structure
Our overall structure is organized like so:
- build : scripts to build the tools needed to run the machine
- dev  : development scripts and files
- doc : documentation, including images present on the wiki and all the reports we wrote while testing the project
- src : source files used to run the program
- tests : testing scripts and files used during development to check
- old.zip : archived files

## Running the code with the GUI
To run the code with the GUI for the first time, install the app to your windows computer using the setup.exe file found in `src/NanoPackUI/bin/publish/setup.exe` and follow the prompts as mentioned in the setup instructions.  Once installed, you should be able to open the app by searching for NanoPackUI  in the windows search bar of the computer.


## Running a headless (GUI-free) run of the code
Within the Control.py file, you will find the following:
```
 20     ## DISABLE USER INTERFACE
 21     headless_mode = False
```
Here headless mode needs to be changed to True to enable running without a GUI.  Be forewarned: this needs to be changed back to False before running the GUI or it will not function properly.
Then you need to open up a terminal by searching for "Anaconda Powershell".  Here use the cd command to navigate your computers file structure until you make it to the src file of the github repository.  Then, verify that you are using the (NanoPackEnv) anaconda enviornment not (base) and if not run the command `conda activate NanoPackEnv`.  Then run the command `python Control.py` and you will be able to run the control script from the terminal without the GUI.

## Location of key files within src
All of the code and its constituent functions are described in detail on the wiki page: [Software Module Overview](https://github.com/BostonUniversitySeniorDesign/NanoView_G33/wiki/Software-Module-Overview).  However, we will give a quick summary of the important files in src here.
- Control.py : contains the top-level code that orchestrates the whole machine from a python perspective.
- loop_functions.py : contains the functions to describe the three main stages that the machine goes through while packng; Initialization Steps, Control Loop, and Closing Steps.
- control_functions.py: contains a number of helper functions which implement functionality the control needs in order to improve the clarity of the control scripts.
- get_chip_img.py: contains the functions to move the machine based on configured parameters and to perform edge detection using computer vision.
- config_distance.yml: contains the configurable parameters most of which are used to move the machine around or convert units.
- chip_error.log: output file containing the information on whatever chips failed for a given run of the machine.
- ./tinyg/tinyg.py: used abstract the communication with the TinyG board away from the end user, so that in control we can just call functions to move the machine.
- ./machine_learning/FindChipsAndNums: contains functions which extract chips from the actuator camera and their respective chip numbers
- ./machine_learning/ExtractDigits: contains functions which extract individual digits from the chip numbers 
- ./machine_learning/digit_detection_pl: contains the pipeline for extracting chip digits and classifying them via our machine learning
- ./machine_learning/Nano_yolo.py: contains functions which run yolov5 and output a useable result. 
- ./machine_learning/yoloML/* : contains all nessisary libraries and dependancies to run yolov5. This includes weights and results. 
- ./helper_scripts/csv_parse.py: functions to parse the CSV in the format that we expect into a list of dictionaries that control can use.
- ./helper_scripts/Pipe.py: python script to control the Win32 pipe we use to communicate with the GUI.
