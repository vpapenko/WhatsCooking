#  What's Cooking? kaggle competition.
#  https://www.kaggle.com/c/whats-cooking-kernels-only
#  
#  Neural network with dense layer using Keras
#  
#  Input: list of ingredients
#  Output: cuisine
#  
#  Author
#  https://www.kaggle.com/vpapenko

import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense

train = pd.read_json('../input/train.json')

t = Tokenizer()
t.fit_on_texts(train['ingredients'])
X = t.texts_to_matrix(train['ingredients'])

encoder = LabelBinarizer()
Y = encoder.fit_transform(train['cuisine'])

model = Sequential()
model.add(Dense(1000, input_dim = len(X[0]), kernel_initializer='glorot_uniform', activation = 'relu'))
model.add(Dense(len(Y[0]), kernel_initializer='glorot_uniform', activation = 'softmax'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, Y, epochs = 1, batch_size = 15)


test = pd.read_json('../input/test.json')
X = t.texts_to_matrix(test['ingredients'])

prediction = model.predict(X)
submission = pd.DataFrame(columns = ['id', 'cuisine'])
submission['id'] = test['id']
submission['cuisine'] = encoder.inverse_transform(prediction)
submission.to_csv('submission.csv', index = False)
print('Submission saved.')
