import os
########################
# HSMM settings
####################
column_names = ['acceleration']
#column_names = ['roll_med_acc_x', 'roll_med_acc_y', 'roll_med_acc_z']

#For 1b_HSSM_batches:
batch_size = 0 #10

# Maximum number of states
Nmax = 4 #10
# Number of resamples to train the model. With more data, you need fewer resamples to converge.
nr_resamples = 20 #15
# Maximum duration of one state (in nr of 5-second intervals). A smaller number speeds up the calculations significantly
truncate = 720 #360 = 30 minutes, 720 = 60 minutes


########################
# Model name 
########################
# The name under which your model is saved
model_name = 'mod_{}st_{}b_{}r_{}t_{}'.format(Nmax,batch_size,nr_resamples,truncate, '_'.join(column_names))

########################
# Data folders
########################
# The folder structure:
# - root path
# ----- accelerometer_5second
# ----- merged
# ----- subsets
# ----- results
# ---------- model_name
# -------------- model.pkl
# -------------- config.py
# -------------- images
# -------------- datawithstates

#This is the root path where your data lives
data_root_path = "/media/sf_VBox_Shared/London/UCL_data_March/run_2017march21/"

#For 0_PrepareData
annotations_path = os.path.join(data_root_path, 'tud.csv')
wearcodes_path = os.path.join(data_root_path, 'wearcodes.csv')
accelerometer_5sec_path = os.path.join(data_root_path, 'accelerometer_5second/')
merged_path = os.path.join(data_root_path, 'merged/')
subset_path = os.path.join(data_root_path, "subsets/")

#For 1_HSMM
train_path = merged_path
results_path = os.path.join(data_root_path, 'results')
model_path = os.path.join(results_path, model_name)
model_file = os.path.join(model_path, 'model.pkl')
states_path = os.path.join(model_path,'datawithstates')
config_file = os.path.join(model_path,'config.py')
image_path = os.path.join(model_path,'images')

#For 2_AnalyseResutls
activities_simplified_path = os.path.join(data_root_path, 'TUD_simplified.csv')


# Create directories:
for pathname in [merged_path, subset_path, results_path, model_path, states_path, image_path]:
    if not os.path.exists(pathname):
        os.makedirs(pathname)


