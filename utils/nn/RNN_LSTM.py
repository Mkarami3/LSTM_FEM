import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

class RNN:

    @staticmethod
    def build(input_size, pretrained_weights = None):

        inputs = Input(input_size)

        conv1 = TimeDistributed(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation = 'relu', padding = 'same')) (inputs)
        padd1 = TimeDistributed(ZeroPadding3D(padding = ( (0, 1), (0, 1), (0, 1)) ))(conv1)
        pool1 = TimeDistributed(MaxPooling3D(pool_size=(3, 3, 3)))(padd1)
        flat = TimeDistributed(Flatten())(pool1)

        # lstm1 = LSTM(1024, return_sequences=True)(flat)
        lsmt2 = Bidirectional(LSTM(512, return_sequences=True))(flat)
        shaping = TimeDistributed(Reshape((4, 4, 2, 32)))(lsmt2)

        up1 = TimeDistributed(UpSampling3D(size = (4, 4, 3)))(shaping)
        padd2 = TimeDistributed(ZeroPadding3D(padding = ( (0, 1), (0, 1), (0, 1)) ))(up1)
        # crop1 = TimeDistributed(Cropping3D(cropping = ( (0, 1), (0, 1), (0, 1)) ))(padd2)
        conv2 = TimeDistributed(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation = 'relu', padding = 'same' ))(padd2)

        merge = concatenate([conv1,conv2], axis = 5)
        output = TimeDistributed(Conv3D(filters=3, kernel_size=(1, 1, 1), activation = 'linear'))(merge)

        model = Model(inputs,output)
        model.summary()
        
        opt = Adam(lr=1e-5)
        # model.compile(loss=RNN.custom_loss(inputs), optimizer=opt, metrics=['mse'])
        model.compile(loss='mse', optimizer=opt, metrics=['mse'])

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        for layer in model.layers:
            print("layer shape",layer.output_shape)

        return model

    #input_data is external force, y_pred is displacement
    def custom_loss(input_data):
        def loss(y_true, y_pred):

            mse_error = K.mean(K.square(y_pred - y_true), axis=-1) #Typical Loss function
            physical_error = RNN.physical_loss_seq(input_data, y_pred, y_true)

            return mse_error + physical_error# regularization value = 0.001

        return loss


    def physical_loss_seq(force, disp_pred, disp_true):

        Batch_size = 4
        zero = tf.constant(0, dtype=tf.float32)
        force_mag = tf.reduce_mean(force, [-1,-2,-3,-4,-5])
        transformed_forces = []
        transformed_forces.append(force[0, :, :, :, :, :])

        for i in range(1,Batch_size):
            pred = tf.cond(tf.reduce_all(tf.equal(force_mag[i], zero)), true_fn = lambda:  force[i-1, :, :, :, :, :],false_fn= lambda: force[i, :, :, :, :, :])
            transformed_forces.append(pred)

        force_augmented = tf.stack(transformed_forces)

        # product_pred = force_augmented * disp_pred
        # disp_true = force_augmented * disp_true
        # mse_error = tf.math.reduce_sum(tf.math.square(product_pred - disp_true), axis=-1)
        
        force_dot_disp = tf.math.reduce_sum(force_augmented * disp_pred, axis = -1)
        return tf.nn.relu(-20*force_dot_disp)
        # return mse_error

    def physical_loss_relu(force, disp):
        '''
        if vector product with disp is positive, then loss=0
        if vector product with disp is negative, then loss=sigmoid(force*disp)
        '''
        force_dot_disp = K.sum(force * disp, axis = -1)
        return K.relu(-20*force_dot_disp)
        
    def physical_loss(force, disp):
        '''
        if vector product with disp is positive, then loss=0
        if vector product with disp is negative, then loss=sigmoid(force*disp)
        '''
        force_dot_disp = K.sum(force * disp, axis = -1)
        is_positive = force_dot_disp > 0

        loss_for_positive = tf.zeros_like(force_dot_disp,dtype='float32')
        loss_for_negative = K.sigmoid(force_dot_disp)

        return tf.where(is_positive, loss_for_positive, loss_for_negative)

