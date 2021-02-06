import numpy as np
import h5py
import json

# print(float(means['F_x']))

class HDF5DatasetGenerator:
    
    def __init__(self, dbPath, batchSize,json_FORCE, json_DISP): 
                 
        self.batchSize = batchSize
        self.db = h5py.File(dbPath)
        self.numFiles = self.db["computed_displacements"].shape[0]

        self.FORCE_MEAN, self.FORCE_STD  = self.dic_to_np(json_FORCE)
        self.DISP_MEAN, self.DISP_STD = self.dic_to_np(json_DISP)
        self.dbtype = dbPath.split("\\")[-1]
        
    def generator(self):

        while True:
            for i in np.arange(0, self.numFiles, self.batchSize):

                # if (self.db["external_forces"][i: i + self.batchSize].shape[0]  != 4) and (self.dbtype != "test.hdf5"):
                #     print("one skipped")
                #     continue

                force = self.db["external_forces"][i: i + self.batchSize]
                displacement = self.db["computed_displacements"][i: i + self.batchSize]

                force_normalize = (force - self.FORCE_MEAN)/self.FORCE_STD
                displacement_normalize = (displacement - self.DISP_MEAN)/self.DISP_STD
                    
                yield (force_normalize, displacement_normalize)#normalize the date by the average

    
    def dic_to_np(self,json_file):

        dic = json.loads(open(json_file).read())

        mean_x = float(list(dic.values())[0])
        mean_y = float(list(dic.values())[1])
        mean_z = float(list(dic.values())[2])

        std_x = float(list(dic.values())[3])
        std_y = float(list(dic.values())[4])
        std_z = float(list(dic.values())[5])       
        # print("{}/{}/{}".format(mean_x, mean_y, mean_z))
        mean_array = np.array([mean_x, mean_y, mean_z]).reshape(1,1,1,1,3)
        std_array = np.array([std_x, std_y, std_z]).reshape(1,1,1,1,3)

        return mean_array, std_array

    def close(self):
        self.db.close()
        
        
        
        
        
        
        