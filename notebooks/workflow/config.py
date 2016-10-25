import os

########################
# Data folders
########################

#This is the root path where your data lives
data_root_path = "/media/sf_VBox_Shared/London/raw/first5/" #next161/"

#For 0_PrepareData
annotations_path = os.path.join(data_root_path, 'tud_first5_deb.csv')
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
#column_names = ['acceleration']
column_names = ['anglex', 'angley', 'anglez', 'acceleration']
#column_names = ['roll_med_acc_x', 'roll_med_acc_y', 'roll_med_acc_z', 'dev_roll_med_acc_x', 'dev_roll_med_acc_y', 'dev_roll_med_acc_z']

# Maximum number of states
Nmax = 4
# Number of resamples to train the model. With more data, you need fewer resamples to converge.
nr_resamples = 30 #15
# Maximum duration of one state (in nr of 5-second intervals). A smaller number speeds up the calculations significantly
truncate = 600

# The name under which your model is saved
model_name = 'model_{}states.pkl'.format(Nmax)
#model_name = 'model_acc_{}states.pkl'.format(Nmax)
#model_name = 'model_roldev_{}states.pkl'.format(Nmax)
states_path_model = os.path.join(states_path, model_name)