import h5py
import os
import numpy as np

class HDF5DatasetWriter:
    
    def __init__(self, dims, outputPath, bufSize=500):

        time_step_index = 1
        x_dim = dims
        y_dim = dims[:time_step_index] + dims[time_step_index+1:] #skip the time step dimension
        
        if os.path.exists(outputPath):
            raise ValueError("The supplied path alreadythe file before continuing.", outputPath)
        
        print("[INFO] Writing external_forces({}) in HDF5 format".format(dims))
        print("[INFO] Writing computed_displacements({}) in HDF5 format".format(dims))

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("external_forces", dims, dtype="float")
        self.labels = self.db.create_dataset("computed_displacements", dims, dtype="float") 
        
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0       
        
    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
            
    
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}
        

        
    def close(self):
        
        if len(self.buffer["data"]) > 0:
            self.flush()
            
        self.db.close()
        
        
        
        
                                                
    