import glob
import os
import numpy as np 
features = ["eeg"] + ["mel"] + ["envelope"] +["wav"]
data_folder = '/ddnstor/imu_liuxin/fyuan/auditoryeegdata/'
data_folder = os.path.join(data_folder)
train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
 
for fpath in train_files:  
        trainpath = fpath
        valpath = fpath[:42]+'val'+fpath[47:]
        testpath = fpath[:42]+'test'+fpath[47:] 
        newdatapath =  fpath[:41]+'new'+fpath[41:] 

        traindata = np.load(trainpath)
        valdata = np.load(valpath)
        testdata = np.load(testpath)
        newdata = np.concatenate((traindata,valdata,testdata),axis=0)
        #print(traindata.shape,valdata.shape,testdata.shape,newdata.shape)
        
        
        np.save(newdatapath,newdata) 
        
        