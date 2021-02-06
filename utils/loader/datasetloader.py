'''
Loading external force and computed displacement from vtk simulation files
'''
import os
class DatasetLoader:
    
    @staticmethod    
    def load(dataset_path):
        '''
        input: data path to vtk simulation files
        return: a list includes path to individual path to each file
        '''
 
        file_paths = []
        for file_path in sorted(os.listdir(dataset_path)):
            
            # print('[INFO] Reading Folder named: {}'.format(folder))
            if file_path.split('.')[-1] == 'vtk':

                full_path = os.path.join(dataset_path, file_path)
                file_paths.append(full_path)
        
        return file_paths

        

    


    
 