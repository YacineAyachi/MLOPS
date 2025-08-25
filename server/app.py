# from fastapi import FastAPI
# from pymongo import MongoClient

# # Create a FastAPI app instance
# app = FastAPI()
# client = MongoClient('db', 27017)
# db = client.test_database
# collection = db.test_collection

# # Define a root endpoint
# @app.get("/")
# def read_root():
#     return {"message": "Hello World"}

# # Function to add fruits to MongoDB
# @app.get("/add/{fruit}")
# def add_list_fruits(fruit):
#     id = collection.insert_one({"fruit": fruit})
#     return list(collection.find({}, {"_id": False}))

# # Route to list all fruits in MongoDB
# @app.get("/list")
# async def list_fruits():
#     return {"results": list(collection.find({}, {"_id": False}))}

from fastapi import FastAPI
from pymongo import MongoClient
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load('model.pkl')

app = FastAPI()

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Mapping des classes
class_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


@app.post("/predict")
def predict(request: PredictionRequest):
    # Extract feature values from the request
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    # Predict the class of the input features
    prediction = model.predict(features)
    
    # Return the prediction
    #return {"prediction": int(prediction[0])}

        # Obtenir le nom de la classe prédite
    predicted_class_name = class_names.get(int(prediction[0]), "Unknown")

    # Retourner le nom de la classe prédite
    return {"prediction": predicted_class_name}

