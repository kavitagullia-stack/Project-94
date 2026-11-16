import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X_train = pd.read_csv("X_train_processed.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test_processed.csv")
y_test = pd.read_csv("y_test.csv")
model = Sequential()
model.add(Dense(8, input_dim=45, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50)
model.evaluate(X_test, y_test)