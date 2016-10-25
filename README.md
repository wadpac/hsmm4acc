# UK Movement Sensing
This repository is for the project by NLeSC in collaboration with UCL, of analyzing movement sensor data of adolescents.

## Installation
Prerequisites:
*  Python 2.7
* pip
* cython. When using conda, you should install cython through conda instead of pip

Open a command line and navigate to the root of this repository. To install, try:

`pip install .`

To run tests:

`nosetests test/`

## Running the workflow
If you have data that is preprocessed by the R-package GGIR, you can use the ipython-workflow to process it.
Configure the `config.py` for your settings. 
