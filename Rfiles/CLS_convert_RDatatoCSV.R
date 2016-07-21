rm(list=ls())
graphics.off()
#==================================================================
# INPUT NEEDED:
setwd("/media/sf_VBox_Shared/London/raw/first5") #<= the folder in which there is a folder named output_...
#===================================================
# Define output directories:
datafolder = "output_raw"
outdir = "accelerometer_5second" # epoch data
outdir2 = "accelerometer_40Hz" # raw data
if (file.exists(outdir) == FALSE) dir.create(outdir)
if (file.exists(outdir2) == FALSE) dir.create(outdir2)

#===================================================
# Define which files need to beprocessed:
ufn = unique(dir(paste0(datafolder, "/meta/raw")))
# Alternatively, only use data from files for which there is a corresponding diary:
# ann = read.csv("D:/dropbox/Dropbox/Accelerometry/GGIR/development/input_cls/data_annotations.csv")
# ufn = as.character(unique(ann$filename))
#===================================================
# Extract identifiers of indiduals from filenames:
myfun = function(x) {
  return(unlist(strsplit(x,"_day"))[1])
}
ufn2 = unique(unlist(lapply(ufn,myfun)))
myfun2 = function(x) {
  return(unlist(strsplit(x,"[.]RDa"))[1])
}
path = paste0(datafolder, "/meta/ms2.out")
fnames = dir(path)
fnames2 = unique(unlist(lapply(fnames,myfun2)))
rpath = paste0(datafolder, "meta/raw")
rfiles = dir(rpath)
rfiles2 = unlist(lapply(rfiles,myfun))

print("Load and export epoch data")
for (i in 1:length(ufn2)) {
  file2read = fnames2[which(fnames2 == ufn2[i])]
  if (length(file2read) > 0) {
    load(paste0(path,"/",file2read,".RData"))
    invalid = IMP$rout[,5]
    invalid = rep(invalid,each=(IMP$windowsizes[2]/IMP$windowsizes[1]))
    NR = nrow(IMP$metashort)
    if (length(invalid) > NR) {
      invalid = invalid[1:NR]
    } else if (length(invalid) < NR) {
      invalid = c(invalid,rep(0,(NR-length(invalid))))
    }
    output = cbind(IMP$metashort,invalid)
    names(output)[2] = "acceleration"
    if (nrow(output) > (1440 * 12)) {
      day1 = output[1:(1440*12),]
      day2 = output[(1440*12+1):nrow(output),]
      write.csv(day1,file=paste0(outdir,"/",file2read,"_day1.csv"),row.names = FALSE)
      write.csv(day2,file=paste0(outdir,"/",file2read,"_day2.csv"),row.names = FALSE)
    } else {
      day1 = output[1:(1440*12),]
      write.csv(day1,file=paste0(outdir,"/",file2read,"_day1.csv"),row.names = FALSE)
    }
  }
}
print("Load and export raw data")
for (i in 1:length(ufn2)) {
  print(i)
  file2read = rfiles[which(rfiles2 == ufn2[i])]
  if (length(file2read) > 0) {
    for (j in 1:length(file2read)) {
      print(paste0(i,".",j))
      load(paste0(rpath,"/",file2read[j]))
      S = cbind(Gx,Gy,Gz) #,temperature,light
      names(S) = c("x","y","z")
      write.table(S,file=paste0(outdir2,"/",file2read[j],".csv"),row.names = FALSE,sep=",")
    }
  }
}
