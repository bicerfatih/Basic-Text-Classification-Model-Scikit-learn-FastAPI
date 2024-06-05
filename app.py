# Hosts the REST API
# Import necessary libraries
import uvicorn
from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# Define a Pydantic model for input validation
class TextData(BaseModel):
    text: str  # Define a required field 'text' of type string

# Create a FastAPI app instance named 'app2'
app = FastAPI()

# Load a pre-trained classifier model from a pickle file
with open("models/classifier.pkl", "rb") as file:
    model = pickle.load(file)  # Deserialize object from file

# Define a route for POST requests to '/predict' that expects TextData
@app.post('/predict')
def make_prediction(data: TextData):
    # Extract the text attribute from the input data
    input_text = data.text
    # Use the loaded model to predict the class of the input text
    prediction = model.predict([input_text])[0]  # Get the first (and only) prediction
    # Return a JSON object containing the prediction
    return {"prediction": prediction}

# Entry point for running the API if the file is executed as the main script
if __name__ == '__main__':
    # Configure and start the uvicorn server to host the API
    uvicorn.run(app, host='127.0.0.1', port=8080)  # Specify host and port

# curl -X POST -H "Content-Type: application/json" -d '{"text": "I hate you so much"}' http://127.0.0.1:8080/predict
