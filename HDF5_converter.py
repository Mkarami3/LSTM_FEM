from utils.config import config
from utils.loader import DatasetLoader
from utils.config import config
from utils.preprocess import Preprocessor
from sklearn.model_selection import train_test_split
from utils.io import HDF5DatasetWriter
import json
import numpy as np
import progressbar

step_size = config.time_step
data_shape = config.data_shape
channel_first=False
paths =  DatasetLoader.load(config.data_path)   

(trainPaths, testPaths)  = train_test_split(paths, test_size=config.NUM_TEST, shuffle=False, random_state=None, stratify=None)               
(trainPaths, valPaths) = train_test_split(trainPaths,test_size=config.NUM_VAL, shuffle=False, random_state=None, stratify=None) 
   
datasets = [
("train", trainPaths, config.TRAIN_HDF5),
("val", valPaths, config.VAL_HDF5),
("test", testPaths, config.TEST_HDF5)]

(F_mean_x, F_mean_y, F_mean_z) = ([], [], [])
(F_std_x, F_std_y, F_std_z) = ([], [], [])
(disp_mean_x, disp_mean_y, disp_mean_z) = ([], [], [])
(disp_std_x, disp_std_y, disp_std_z) = ([], [], [])

for (dType, paths, outputPath) in datasets:

    print("[INFO] building {}...".format(outputPath))
    dim_0 = data_shape[0]
    dim_1 = data_shape[1]
    dim_2 = data_shape[2]
    dim_3 = data_shape[3]
    writer = HDF5DatasetWriter((len(paths),step_size, dim_0, dim_1, dim_2, dim_3), outputPath) #Number of cells=1782 in 3 directions (x,y,z)
    
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()

    for i in range(0, len(paths), step_size):

        forces_time_steps = np.zeros((step_size, data_shape[0], data_shape[1], data_shape[2],data_shape[3]))
        disps_time_steps = np.zeros((step_size, data_shape[0], data_shape[1], data_shape[2],data_shape[3]))
        meshfiles = paths[i:i+step_size]

        for (k,meshfile) in enumerate(meshfiles):
            force, disp, force_mean, disp_mean, force_std, disp_std = Preprocessor.array_reshape(meshfile, data_shape,channel_first)
            forces_time_steps[k, :, :, :, :] = force
            disps_time_steps[k, :, :, :, :] = disp

            F_mean_x.append(force_mean[0])
            F_mean_y.append(force_mean[1])
            F_mean_z.append(force_mean[2])
            F_std_x.append(force_std[0])
            F_std_y.append(force_std[1])
            F_std_z.append(force_std[2])

            disp_mean_x.append(disp_mean[0])
            disp_mean_y.append(disp_mean[1])
            disp_mean_z.append(disp_mean[2])
            disp_std_x.append(disp_std[0])
            disp_std_y.append(disp_std[1])
            disp_std_z.append(disp_std[2])

   
        writer.add([forces_time_steps], [disps_time_steps])
        pbar.update(i)
    
    pbar.finish()
    writer.close()                        

print("[INFO] serializing means...")
FORCE = {"F_mean_x": str(np.mean(np.abs(F_mean_x))), "F_mean_y": str(np.mean(np.abs(F_mean_y))), "F_mean_z": str(np.mean(np.abs(F_mean_z))), 
        "F_std_x": str(np.max(F_std_x)), "F_std_y": str(np.max(F_std_y)), "F_std_z": str(np.max(F_std_z))}

DISP = {"disp_mean_x": str(np.mean(np.abs(disp_mean_x))), "disp_mean_y": str(np.mean(np.abs(disp_mean_y))), "disp_mean_z": str(np.mean(np.abs(disp_mean_z))), 
        "disp_std_x": str(np.max(disp_std_x)), "disp_std_y": str(np.max(disp_std_y)), "disp_std_z": str(np.max(disp_std_z))}

print(FORCE)
print(DISP)
f = open(config.FORCE_MEAN, "w")
f.write(json.dumps(FORCE))
f.close()

f = open(config.DISP_MEAN, "w")
f.write(json.dumps(DISP))
f.close()        