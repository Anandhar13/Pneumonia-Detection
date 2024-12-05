from tensorflow.keras.optimizers import Adam # import adam optimizer
from keras.models import Sequential #Import Sequential model API
from keras.layers import (Dense,Conv2D,MaxPooling2D, Dropout,BatchNormalization,Flatten,MaxPool2D) #IMporting required layers for CNN


class PnemoniaDetection:

    def __init__(self,input_shape=(256,256,3),learning_rate=0.001):
        """
         INtialize the model with the above input shape and learning rate

        :param input_shape: dimension of ininput shape in tuple data structure
        :param learning_rate: learning rate for optimizer
        """

        self.input_shape=input_shape
        self.learning_rate=learning_rate

        self.model = self.build_model() #build the CNN model

    def build_model(self):

        """
        Bulid and complie the CNN mode
        :returns the complied sequential model
        """

        model = Sequential() # Initialize Sequential model

        #First Convluation 2D layer with 32 filters to extract features
        model.add(Conv2D(32,(3,3),strides = 1, padding = 'same', activation = 'relu', input_shape=self.input_shape))
        model.add(BatchNormalization()) #Normalize activations
        model.add(MaxPool2D(2,2),strides=2,padding = 'same') #Downsampling with max pooling


        # Second Conv2D layer: Increase feature maps to 64
        model.add(Conv2D(64,(3,3),strides = 1, padding = 'same', activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1)) #deactive some percentage of neuron to prevent overfitting
        model.add(MaxPool2D((2,2),strides=2,padding = 'same'))

        #Third conv2D layer: Add another 64 filters to learn more features
        model.add(Conv2D(64,(3,3),strides = 1, padding = 'same', activation = 'relu'))
        model.add(BatchNormalization())  # Normalize activations
        model.add(MaxPool2D((2, 2), strides=2, padding='same'))

        #Fourth COnv2D layer Increase feature maps to 128
        model.add(Conv2D(128,(3,3),strides = 1, padding = 'same', activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(MaxPool2D((2, 2), strides=2, padding = 'same'))

        #Fifth Conv2D layer: increase feature maps to 256
        model.add(Conv2D(256,(3,3),strides = 1, padding = 'same', activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(MaxPool2D((2, 2), strides=2, padding = 'same'))


        
