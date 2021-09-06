# Mullan Item Checker

AI Model using YOLO (You Only Look Once) to detect items and check on the data base which of them are needed.

## Installation

You will need to use [pip](https://pip.pypa.io/en/stable/) to the item checker.
It has a few depedencies to use YOLO. You will need to use [TrainYourOwnYOLO](https://github.com/AntonMu/TrainYourOwnYOLO), here is the steps to install everything (assuming you have Python installed):

```bash
git clone https://github.com/AntonMu/TrainYourOwnYOLO
cd TrainYourOwnYOLO/
```
### On (mac/linux) environment
```bash
python3 -m venv env
source env/bin/activate
```
### On Windows environment
You will need to open a PowerShell (might need to be admin in order to create the virtual environment.
```bash
py -m venv env
.\env\Scripts\activate
```

### Installing packages for Windows/Linux/Mac
```bash
pip install pip --upgrade
pip install -r requirements.txt
pip install Pillow pyodbc opencv-contrib-python
```

You will then need to download the files from this github repository and put them into "TrainYourOwnYOLO\3_Inference". You should have the basic Detector.py file provided by TrainYourOwnYOLO and the two files from this repository. 

It is strongly advised to use **CuDNN/CUDA** for faster implementation, but check the [compatibility](https://www.tensorflow.org/install/source?hl=fr#gpu) first. We won't go thru the installation of CUDA here because it would be too long.

## Usage

You will need to change line **174 - 177** and **192 - 201** to specify the credentials.

To run the program you will need to go in the folder where the code is situated using the cd command and just run it by using ```py Semi``` then press ```TAB``` to have the correct path name. 

## Hikvision program
This is a small program to test the hikvision camera and see it working, *or not*. You just have to specify the credentials on **line 10** of the python code and run it. 
