NUM_VAL = 300
NUM_TEST = 10


data_shape = (17, 17, 7, 3)
time_step = 2
data_shape1 = (time_step,17, 17, 7, 3)
# dataset path for reading vtk files
data_path = "D:\\Karamim\\dataset\\"

# define the path to the output directory used for storing plots,
TRAIN_HDF5 = "D:\\Karamim\\RNN_FEM\\utils\\HDF5_files\\train.hdf5"
VAL_HDF5 = "D:\\Karamim\\RNN_FEM\\utils\\HDF5_files\\val.hdf5"
TEST_HDF5 = "D:\\Karamim\\RNN_FEM\\utils\\HDF5_files\\test.hdf5"
MODEL_PATH = "D:\\Karamim\\RNN_FEM\\utils\\HDF5_files\\model.h5"
OUTPUT_PATH = "D:\\Karamim\\RNN_FEM\\utils\\HDF5_files\\"


FORCE_MEAN = "D:\\Karamim\\RNN_FEM\\utils\\HDF5_files\\FORCE_MEAN.json"
DISP_MEAN = "D:\\Karamim\\RNN_FEM\\utils\\HDF5_files\\DISP_MEAN.json"

