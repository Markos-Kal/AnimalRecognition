from tensorflow import keras

model = keras.models.load_model('animal_classifier_model.h5')
model.save('model.keras', save_format='keras')