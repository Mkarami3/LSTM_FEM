from utils.config import config
from utils.loader import DatasetLoader
from utils.io import HDF5DatasetGenerator
from utils.nn import RNN
from utils.monitor import TrainingMonitor
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

epoch = 100
print("[INFO] initialize data generator...")
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 8, config.FORCE_MEAN, config.DISP_MEAN)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 8, config.FORCE_MEAN, config.DISP_MEAN)      
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.NUM_TEST, config.FORCE_MEAN, config.DISP_MEAN)    

print("[INFO] compiling model...")
model = RNN.build(input_size=config.data_shape1)

path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

model = load_model(config.MODEL_PATH,compile=False)
model.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['mse'])

H = model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numFiles // 8,
    validation_data=valGen.generator(),
    validation_steps=valGen.numFiles // 8,
    epochs=epoch,
    callbacks=callbacks, verbose=1
    )

print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)
trainGen.close()
valGen.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["mse"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_mse"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.show()


