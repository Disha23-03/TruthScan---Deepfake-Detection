from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

print("Running from:", os.getcwd())

model = tf.keras.models.load_model("deepfake_model.h5")

IMG_SIZE = (224, 224)

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file)

        processed = preprocess(image)
        prediction = model.predict(processed)[0][0]

        result = "AUTHENTIC" if prediction > 0.5 else "DEEPFAKE"
        confidence = float(prediction * 100)

        return jsonify({
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)