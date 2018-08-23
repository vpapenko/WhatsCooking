import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

train = pd.read_json('./train.json')
train = train.sample(frac=1).reset_index(drop=True)
#train = train.loc[0 : 5000]

t = Tokenizer()
t.fit_on_texts(train['ingredients'])
X = t.texts_to_matrix(train['ingredients'])

encoder = LabelBinarizer()
Y = encoder.fit_transform(train['cuisine'])

trainCount = int(len(X) * 0.9)
testCount =  len(X) - trainCount

Xtrain = X[0 : trainCount]
Ytrain = Y[0 : trainCount]
Xtest = X[trainCount + 1 : trainCount + 1 + testCount]
Ytest = Y[trainCount + 1 : trainCount + 1 + testCount]

model = Sequential()
model.add(Dense(5000, input_dim = len(X[0]), kernel_initializer='glorot_uniform', activation = 'relu'))
model.add(Dense(len(Y[0]), kernel_initializer='glorot_uniform', activation = 'softmax'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, Y, epochs = 2, batch_size = 15, verbose = 1)

prediction = model.predict(Xtest)

print('')

wrong = 0
for i in range(len(Xtest) - 1):
    if np.argmax(prediction[i]) != np.argmax(Ytest[i]):
        wrong += 1
print('')
print(str(wrong) + ' in ' + str(len(Xtest)))
print(str(1 - wrong/len(Xtest)))
print('')