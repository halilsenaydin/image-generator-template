from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

model = load_model(r'Model Path')
root = r'Dataset Path'

targetSize = (224,  224)
batchSize = 1
classMode = 'binary' # 'categorical'

testDataGen = ImageDataGenerator()
testDatas = testDataGen.flow_from_directory(f'{root}\\Test Datas',
                                            target_size=targetSize,
                                            batch_size=batchSize,
                                            class_mode=classMode)

evaluate  = model.evaluate_generator(testDatas)
print(f'Loss: {evaluate[0]} :: Accuracy: {evaluate[1]}')

testDatas.reset()

pred = model.predict_generator(testDatas, verbose=1)

# Binary?
pred[pred > .5] = 1
pred[pred <= .5] = 0

labels = testDatas.labels
fileNames = testDatas.filenames

# Results
result = pd.DataFrame()
result['fileNames'] = fileNames
result['testLabels'] = labels
result['predicts'] = pred
result.to_excel('result.xlsx')