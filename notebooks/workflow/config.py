import os

########################
# Data folders
########################

#This is the root path where your data lives
data_root_path = "/media/windows-share/London/data_13012017/run_20170307/"

#For 0_PrepareData
annotations_path = os.path.join(data_root_path, 'tud.csv')
wearcodes_path = os.path.join(data_root_path, 'wearcodes.csv')
accelerometer_5sec_path = os.path.join(data_root_path, 'accelerometer_5second/')
merged_path = os.path.join(data_root_path, 'merged/')
subset_path = os.path.join(data_root_path, "subsets/")

#For 1_HSMM
model_path = os.path.join(data_root_path, 'models')
states_path = os.path.join(data_root_path,'datawithstates')

#For 2_AnalyseResutls
activities_simplified_path = os.path.join(data_root_path, 'TUD_simplified.csv')

########################
# HSMM settings
####################
#column_names = ['acceleration']
column_names = ['acceleration'] # ,
#column_names = ['roll_med_acc_x', 'roll_med_acc_y', 'roll_med_acc_z']

#For 1b_HSSM_batches:
batch_size = 10

# Maximum number of states
Nmax = 10
# Number of resamples to train the model. With more data, you need fewer resamples to converge.
nr_resamples = 15 #15
# Maximum duration of one state (in nr of 5-second intervals). A smaller number speeds up the calculations significantly
truncate = 720 #360 = 30 minutes, 720 = 60 minutes

# The name under which your model is saved
model_name = 'model_{}states_{}batch_{}resamples_{}truncate.pkl'.format(Nmax,batch_size,nr_resamples,truncate)
#model_name = 'model_acc_{}states.pkl'.format(Nmax)
#model_name = 'model_roldev_{}states.pkl'.format(Nmax)
states_path_model = os.path.join(states_path, model_name)
