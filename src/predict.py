import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict(img_path):
    model = load_model("../models/waste_classifier.h5")

    with open("../models/class_indices.json") as f:
        class_idx = json.load(f)

    inv_class_idx = {v:k for k,v in class_idx.items()}

    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0]
    idx = np.argmax(pred)
    label = inv_class_idx[idx]

    print("Predicted Class:", label)
