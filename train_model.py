import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Data
data = []
labels = []

for file in os.listdir("data"):
    if file.endswith(".npy"):
        label = file.split("_")[0]
        landmarks = np.load(os.path.join("data", file))
        data.append(landmarks)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Encode Labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Save label encoder
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2)

# Build Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save('sign_model.h5')
