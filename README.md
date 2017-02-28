# UK Movement Sensing
This repository is for the project by NLeSC in collaboration with UCL, of analyzing movement sensor data of adolescents. It contains code for processing accelerometer data using Hidden Semi Markov models (using the [pyhsmm](https://github.com/mattjj/pyhsmm) package). This software is build for the [Millenium Cohort study](http://www.cls.ioe.ac.uk/page.aspx?sitesectionid=851).

## Installation
Prerequisites:
*  Python 2.7
* pip

### Instalation with conda
Navigate to the root of this repository. Create a conda environment with the environment.yml file:
 
`conda env create -f environment.yml`

Activate the environment:

`source activate ucl`

Then install the package:

`pip install .`

### Installation with pip
Navigate to the root of this repository. To install, try:

`pip install .`

### Running tests
To run tests:

`nosetests test/`

In order to run the 3-d visualizations (in [this notebook](https://github.com/NLeSC/UKMovementSensing/blob/master/notebooks/workflow/3_Visualization.ipynb) ) you need to install [mayavi](http://docs.enthought.com/mayavi/mayavi/installation.html).

## Running the workflow
For processing data of the Millenium Cohort studies, we created a flow of [Jupyter notebooks](http://jupyter.org) that can be found in the [notebooks/workflow directory](https://github.com/NLeSC/UKMovementSensing/tree/master/notebooks/workflow). 
Input for this workflow is:
* accelerometer data that is processed completely with the R-package GGIR into 5-second aggregated data
* A file for diary annotations, and wearcode file for joining the datasets
Edit the `config.py` file to specify the directories for input and output. Also specific settings for the HSMM can be set in this file.

The workflow consists of the following steps:
* **0.Prepare data** Joins the accelerometer with the diary data and performs some basic checks. 
* **1.HSMM** Fits an HSMM model on the data and saves the data with the corresponding states for each time window
* **1b.HSMM batches** If the dataset is very large, this notebook can be used to process the data in batches
* **2.Analyze results** Makes comparison with diary and gives some statistics, to aid interpretation of the states
* **3.Visualize Model** Creates a 3d visualization of the states

The output from step 1 (the data with corresponding states) can be used for further processing, such as aggregating per individual.
