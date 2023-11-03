from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
import numpy as np
import requests

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO


app = Flask(__name__)
api = Api(app)

# Load the pre trained model
pretrained_model = InceptionV3(weights="imagenet")


client = MongoClient("mongodb://db:27017")
db = client.ImageRecognition
users = db["Users"]


def user_exists(username):
    if not users.count_documents({"Username": username}):
        return False
    return True


class Register(Resource):
    """
    docstring for Register
    """

    def post(self):
        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["password"]

        if user_exists(username):
            retJson = {
                "status": 301,
                "msg": "Invalid username, user already exists"
            }
            return jsonify(retJson)

        hashed_pw = bcrypt.hashpw(password.encode("utf8"), bcrypt.gensalt())

        users.insert_one({
            "Username": username,
            "Password": hashed_pw,
            "Tokens": 6
        })

        retJson = {
            "status": 200,
            "msg": "You have successfully signed up for th API"
        }

        return jsonify(retJson)


def verify_pw(username, password):
    if not user_exists(username):
        return False

    hashed_pw = users.find({
        "Username": username
    })[0]["Password"]
    
    if bcrypt.hashpw(password.encode("utf8"), hashed_pw) == hashed_pw:
        return True
    else:
        return False

def verify_credentials(username, password):
    if not user_exists(username):
        return generate_return_dictionary(301, "Invalid username"), True

    correct_password = verify_pw(username, password)

    if not correct_password:
        return generate_return_dictionary(301, "Invalid password"), True

    return generate_return_dictionary(200, "Successful user verification"), False

def generate_return_dictionary(status, msg):
    ret_json = {
        "status": status,
        "msg": msg
    }
    return ret_json

class Classify(Resource):
    def post(self):
        # Get posted data
        postedData = request.get_json()

        # Get credentials and url
        username = postedData["username"]
        password = postedData["password"]
        url = postedData["url"]

        # Verify credentials
        ret_json, error = verify_credentials(username, password)
        if error:
            return jsonify(ret_json)

        # Check if user has tokens
        tokens = users.find({
            "Username": username
        })[0]["Tokens"]

        if tokens <= 0:
            return jsonify(generate_return_dictionary(303, "Not enough tokens"))

        # Classify the image
        if not url:
            return jsonify(({"error": "No url provided"}), 400)

        ## Load image from url
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        ## Pre process the image
        img = img.resize((299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        ## Make prediction
        prediction = pretrained_model.predict(img_array)
        actual_prediction = imagenet_utils.decode_predictions(prediction, top=5)

        # Return classification response
        ret_json = {}
        for pred in actual_prediction[0]:
            ret_json[pred[1]] = float(pred[2]*100)

        ## Reduce tokens
        users.update_one({
            "Username": username
        },{
            "$set": {
                "Tokens": tokens - 1
            }
        })

        return jsonify(ret_json)


class Refill(Resource):
    def post(self):
        # Get posted data
        postedData = request.get_json()

        # Get credentials and url
        username = postedData["username"]
        password = postedData["admin_pw"]
        amount = postedData["amount"]

        # Check if an user exists
        if not user_exists(username):
            return jsonify(generate_return_dictionary(301, "Invalid username"))

        # Check admin password
        correct_pw = "201080"
        if not password == correct_pw:
            return jsonify(generate_return_dictionary(302, "Incorrect password"))

        # Update tokens and respond
        users.update_one({
            "Username": username
        }, {
            "$set": {
                "Tokens": amount
            }
        })

        return jsonify(generate_return_dictionary(200, "Refilled"))


api.add_resource(Register, '/register')
api.add_resource(Classify, '/classify')
api.add_resource(Refill, '/refill')


if __name__=="__main__":
    app.run(host='0.0.0.0')
