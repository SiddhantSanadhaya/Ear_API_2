from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json
import tensorflow as tf
from keras.models import load_model
import constant_values
import cv2
app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class IrisClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        # print(type(X),X.shape)
        X = X.astype(np.uint8)
        img=cv2.resize(X, (constant_values.resize_x,constant_values.resize_y))
        img = img.reshape(constant_values.n_image,constant_values.resize_x,constant_values.resize_y,constant_values.channel)
        prediction = model.predict(img)
        return jsonify(prediction.tolist())

api.add_resource(IrisClassifier, '/iris')

if __name__ == '__main__':
    # Load model
    # with open('model.pickle', 'rb') as f:
    #     model = pickle.load(f)
    with tf.device('/cpu:0'):
      model = load_model(r'C:\Users\Siddhant Sanadhaya\Downloads\model_13_4_22_i2.h5')
    # return model


    app.run(debug=True)