
from pneumonia_detection_model import PneumoniaDetectionModel
from keras.src.legacy.preprocessing.image import ImageDataGenerator


#dataset paths

train_dir = 'dataset/train'
test_dir = 'dataset/test'
val_dir = 'dataset/val'


#train datagenerators for rescaling and data argumentaion
train_datagen = ImageDataGenerator(
    rescale=1./255, #Normalize pixel values to [0, 1]
    shear_range=0.2, # # Random shearing transformations
    zoom_range=0.2, # zooming transformations
    horizontal_flip=True #flip images horizontally
)

#test datagen
test_datagen = ImageDataGenerator(rescale=1./255)

#val datagen
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256), #resize images
    batch_size=32,
    class_mode='binary' #binary classification
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

#create instance of the Pnemonia model
pneumonia_model= PneumoniaDetectionModel()


#train the model using the training and val data generator
history = pneumonia_model.train(
    train_generator = train_generator,
    val_generator=val_generator,
    steps_per_epoch=163 ,
    validation_steps = 624,
    epochs=50
)

#evaluate the model
test_accuracy = pneumonia_model.evaluate(test_generator,steps=624)
train_accuracy = pneumonia_model.evaluate(train_generator,steps=163)

#prin the avaluation results
print('The testing accuracy is: ', test_accuracy[1]*100,'%')
print('The training accuracy is: ', train_accuracy[1]*100,'%')

#Save the model to the specified directory
pneumonia_model.save_model('models/pnemonia_model.h5')
