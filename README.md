# HSMMs for accelerometer data
This repository is for the project by NLeSC in collaboration with UCL, of analyzing movement sensor data of adolescents. It contains code for processing accelerometer data using Hidden Semi Markov models (using the [pyhsmm](https://github.com/mattjj/pyhsmm) package).
This software is meant for accelerometer data processed with the [R-package GGIR](https://github.com/wadpac/GGIR).

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

### Installation troubleshooting
#### ImportError: No module named hmm_messages_interface
The `pyhsmm` package needs the right gcc compiler (it seems to work with gcc 4.7).  You can clone the pyhsmm package and compile it:

`python setup.py build_ext`

Which should solve the issue.
See also https://github.com/mattjj/pyhsmm/issues/55. 

#### Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.
You can disable the use of mkl with:
`conda install nomkl`



### Running tests
To run tests:

`nosetests test/`


