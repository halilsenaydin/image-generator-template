import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
import tensorflow.keras.optimizers as optimizer

inputShape = (224, 224, 3)
targetSize = (224, 224)
batchSize = 1
classMode = 'binary' # 'categorical'
root = r'DataSet Path'

# Create A Model
model = models.Sequential()
model.compile(optimizer=optimizer.Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])

# Load Dataset
from keras.preprocessing.image import ImageDataGenerator
# Bu kütüphane klasördeki tüm resimleri ram'a yüklemiyor. Sırası geleni işliyor.

trainingDataGen = ImageDataGenerator()
testDataGen = ImageDataGenerator()

trainingDatas = trainingDataGen.flow_from_directory(f'{root}\\Training Datas',
                                                 target_size=targetSize,
                                                 batch_size=batchSize,
                                                 class_mode=classMode)

testDatas = testDataGen.flow_from_directory(f'{root}\\Test Datas',
                                            target_size=targetSize,
                                            batch_size=batchSize,
                                            class_mode=classMode)

# Training Model And Save Best Model
epochs =  10
version = 'V1.0'
modelPath = f'.\\SavedModels\\{epochs}EpochsGenderRecognitionModel{version}'
checkPointer = ModelCheckpoint(filepath=f'{modelPath}.h5', verbose=1, save_best_only=True)

history = model.fit_generator(trainingDatas, validation_data=testDatas, epochs=epochs, callbacks=[checkPointer])

# Show Loss And Accuracy Graphic
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Training', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], color ='r', label='Training Loss')
plt.plot(history.history['val_loss'], color ='b', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], color ='g', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color ='m', label='Validation Accuracy')
plt.legend(loc='lower right')

plt.savefig(f'{modelPath}.png')