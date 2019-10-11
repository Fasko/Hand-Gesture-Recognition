'''
# Michael Fasko Jr & Jacob Calfee
# Cleveland State University
# Hackron 4k (10/5/19 - 10/6/19) - Hand Gesture Recognition with Machine Learning

 # File is responsible for taking input of training data from a CSV file and passing into TensorFlow Keeras CNN for
    9 different classifications of hand gesture recognition
'''

import time
import numpy
from keras.layers import Dense, Dropout
from keras.models import Sequential

training_data = []
seed = 7
numpy.random.seed(seed)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# Load CSV dataset with X (training data) and Y (label)
raw_dataset = numpy.loadtxt("right_hand_dataset_extended.csv", delimiter = ",") # right_hand_dataset_reduced10.csv
# right_hand_dataset
X = raw_dataset[:, 0:42]  # Get the first 42 numbers on the line (the coordinates)
Y = raw_dataset[:, 42]  # Get the labels
transformer = Normalizer().fit(X)
X = transformer.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.10, random_state=0) # Randomize Test/Train Splits

# Model
model = Sequential()
model.add(Dense(64, input_dim=(42), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim = 9, activation='softmax')) # init='RandomNormal',
model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy']) # OK
model.fit(X_train, Y_train, epochs=85, batch_size=32, verbose=2,
          validation_data=(X_test, Y_test), shuffle=True)

# Evaluate Model
results = model.evaluate(X_test, Y_test, batch_size=32)
print('test loss, test acc:', results)

named_tuple = time.localtime()
time_string = time.strftime("%m_%d_%Y_%H_%M_%S", named_tuple)

# Save model w/ timestamp and name
model.save("normalized_epochs85_42_data_points_extended_9_outputs" + time_string + ".h5")
