library(mhsmm)

filepath = "/media/sf_VBox_Shared/London/example_data"
filename = paste0(filepath, '/example_bobby_feature5min.csv')
data_10min = read.table(filename, header=TRUE, sep=",")



model <- hmmspec(init, P, parms.emis = B, dens.emis = dpois.hsmm)