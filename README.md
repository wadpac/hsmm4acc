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
`nosetests tests/`

In order to run the 3-d visualizations (in [this notebook](https://github.com/NLeSC/UKMovementSensing/blob/master/notebooks/workflow/3_Visualization.ipynb) ) you need to install [mayavi](http://docs.enthought.com/mayavi/mayavi/installation.html).

## Running the workflow
If you have data that is preprocessed by the R-package GGIR, you can use the ipython-workflow to process it.
Configure the `config.py` for your settings. 