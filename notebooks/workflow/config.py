import os

########################
# Data folders
########################

#This is the root path where your data lives
data_root_path = "/media/sf_VBox_Shared/London/raw/next161/" #first5/"

#For 0_PrepareData
annotations_path = os.path.join(data_root_path, 'tud_next161_deb.csv')
wearcodes_path = os.path.join(data_root_path, 'wearcodes.csv')
accelerometer_5sec_path = os.path.join(data_root_path, 'accelerometer_5second/')
merged_path = os.path.join(data_root_path, 'merged/')
subset_path = os.path.join(data_root_path, "subsets/")

#For 1_HSMM
model_path = os.path.join(data_root_path, 'models')

states_path = os.path.join(data_root_path,'datawithstates')

########################
# HSMM settings
####################
column_names = ['anglex', 'angley', 'anglez', 'acceleration']
Nmax = 4
nr_resamples = 2 #15
truncate = 600