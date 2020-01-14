from tensorflow import keras
from keras_preprocessing import  image
import numpy as np
import wikiquotes

model = keras.applications.resnet.ResNet50(include_top=True, weights='imagenet')

img = image.load_img('photo_id_john.png', target_size=(224,224,3))
img = np.expand_dims(img, axis=0)

predictions = model.predict(img)

predicted_classes = keras.applications.resnet.decode_predictions(predictions, top=2)

string_search = ""

for pred in predicted_classes[0]:
    string_search += pred[1] + " "

print(string_search)

quote = wikiquotes.random_quote(string_search, 'english')

print(quote)
# print(model.summary())