import torch
from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS
import processing_img
from MNIST_model.LeNet import LeNet

app = Flask(__name__)
model = LeNet()
model.load_state_dict(torch.load("../MNIST_model/best_mod.pth"))


CORS(app, resources={'/image': {"origins": "http://localhost:63342"}})


@app.route('/image', methods=['POST'])
def image_post_request():
    x = processing_img.processing(request.json['image'])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y = model(x_tensor.reshape(1, 1, 28, 28)).reshape((10,))
    y = y.detach().numpy().reshape(-1)
    n = int(np.argmax(y, axis=0))
    y = [float(i) for i in y]
    return jsonify({'result':y, 'digit':n})

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=5000)