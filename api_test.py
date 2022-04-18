import numpy as np
import requests
import json
import cv2


# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# Load data
# iris = load_iris()
# data_testing = np.load("datos_testing.npz")['arr_0'].reshape([-1,247,273,3]).astype(np.uint8)
# data_testing = data_testing/np.max(data_testing)
data_testing = cv2.imread(r"C:\Users\Siddhant Sanadhaya\Downloads\Datos\Datos\Training-validation\Earwax plug\e2.jpg")
# data_testing = data_testing.reshape(data_testing.shape[0], 273,247,3)

# Split into train and test sets using the same random state
# X_train, X_test, y_train, y_test = \
#     train_test_split(iris['data'], iris['target'], random_state=12)

# Serialize the data into json and send the request to the model
payload = {'data': json.dumps(data_testing.tolist())}
y_predict = requests.post('http://127.0.0.1:5000/iris', data=payload).json()

# Make array from the list
y_predict = np.array(y_predict)
print(y_predict)