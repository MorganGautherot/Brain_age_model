from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.layers import ReLU, PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint


# block of Cole model
def block_model_Cole(model, nb_block):
    model.add(Conv3D(filters=8*pow(2,nb_block),
                     kernel_size=(3, 3, 3),
                     padding='same',
                     activation='relu',
                     name="conv_"+str(nb_block+1)+"_1"))
    model.add(Conv3D(filters=8*pow(2,nb_block),
                     kernel_size=(3, 3, 3),
                     padding='same',
                     activation=None,
                     name="conv_"+str(nb_block+1)+"_2"))
    model.add(BatchNormalization(name="bn_"+str(nb_block+1)+"_3", axis=1))
    model.add(ReLU(name="relu_conv_"+str(nb_block+1)+"_4"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           strides=(2, 2, 2)))
    return model

# Model
def model_Cole(size):
    """
    Inputs:
        - model_options:
        - weights_path: path to where weights should be saved
    Output:
        - nets = list of NeuralNets (CNN1, CNN2)def Unet_3D_model(modalities, patch_size, filters=32, dropout_rate=0.2):
    """
    model = Sequential()
    model.add(Conv3D(filters=8,
                     kernel_size=(3, 3, 3),
                     padding='same',
                     activation='relu',
                     input_shape=size,
                     name="conv_1_1"))
    model.add(Conv3D(filters=8,
                     kernel_size=(3, 3, 3),
                     padding='same',
                     activation=None,
                     name="conv_1_2"))
    model.add(BatchNormalization(name='bn_1_3', axis=1))
    model.add(ReLU(name='relu_conv_1_4'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           strides=(2, 2, 2)))
    model = block_model_Cole(model, 1)
    model = block_model_Cole(model, 2)
    model = block_model_Cole(model, 3)
    model = block_model_Cole(model, 4)
    
    model.add(Flatten())
    model.add(Dense(units=1,
                    name="d_6"))
    return model


