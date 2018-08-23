import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

seed = 7
np.random.seed(seed)

train = pd.read_json('./train.json')
train = train.sample(frac=1).reset_index(drop=True)
train = train.loc[0 : 5000]

t = Tokenizer()
t.fit_on_texts(train['ingredients'])
X = t.texts_to_matrix(train['ingredients'])

encoder = LabelBinarizer()
Y = encoder.fit_transform(train['cuisine'])

# baseline model
def create_model(optimizer='adam', init='glorot_uniform', activation_hiden = 'relu', activation_out = 'softmax',loss = 'binary_crossentropy',l1=6000,l2=0):
	X_len = 3469#6715
	Y_len = 20
	# create model
	model = Sequential()
	model.add(Dense(l1, input_dim=X_len, kernel_initializer=init, activation = activation_hiden))
	if l2>0:
		model.add(Dense(l2, kernel_initializer=init, activation = activation_hiden))
	model.add(Dense(Y_len, kernel_initializer=init, activation = activation_out))
	model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])
	return model

model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['adam']
init = ['glorot_uniform']
epochs = [1,5,10]
batches = [15]
activation_hiden = ['relu']#, 'sigmoid', 'softmax']
activation_out = ['softmax']#, 'relu', 'sigmoid']
loss = ['binary_crossentropy']
l1=[8000]
l2=[0]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, activation_hiden=activation_hiden, activation_out=activation_out,loss=loss,l1=l1,l2=l2)
param_grid = dict(l1=l1,l2=l2)
param_grid = dict(epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=10)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))