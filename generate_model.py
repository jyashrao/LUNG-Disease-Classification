import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Bidirectional
from tensorflow.keras.models import Model

# Constants from your original code
CLASSES = ["COPD", "Asthma", "Pneumonia", "URTI", "Healthy", "Bronchiectasis", "LRTI", "Bronchiolitis"]
NUM_FEATURES = 52
MAX_FRAMES = 100

def build_model():
    inputs = Input(shape=(MAX_FRAMES, NUM_FEATURES))
    x = Conv1D(128, 5, activation='relu')(inputs)
    x = Bidirectional(GRU(64))(x)
    outputs = Dense(len(CLASSES), activation='softmax')(x)
    return Model(inputs, outputs)

# Generate and save
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save("respiratory_model.h5")

print("Success! 'respiratory_model.h5' has been generated in your project folder.")