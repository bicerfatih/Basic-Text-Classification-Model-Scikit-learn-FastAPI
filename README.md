# Basic-Text-Classification-Model-Scikit-learn-FastAPI
Developing a straightforward text classification model for detecting hate speech using Scikit-learn, and deploying the model with FastAPI.

## Introduction
This documentation details the features and functionalities of different components within my text classification project pipeline. The primary objective of this project is to manage text data efficiently, from the initial stages of data processing to the final prediction output. I aim to develop a fundamental text classification model designed to distinguish between 'hate' and 'noHate' speech within the provided dataset. The model needs to be deployed as a REST API using FastAPI, allowing users to send text input and receive predictions.

## Requirements
Required Python packages include FastAPI, uvicorn, numpy, pandas, scikit-learn, tensorflow, and others as listed in the `requirements.txt`. These packages support the various stages of our machine learning pipeline.

## Approach

1. **Dataset Understanding and Preparation**
   - **Data Source**: The dataset used for training and testing is available [here](https://github.com/Vicomtech/hate-speech-dataset/tree/master).
   - **Data Structure**: The dataset consists of text samples labeled as either `hate` or `noHate`.
   - **Data Loading**: Load the data using pandas and perform basic preprocessing such as tokenization and lowercasing.

2. **Model Training**
   - **Choice of Framework**: The model is implemented using PyTorch for its flexibility and extensive support for deep learning.
   - **Model Architecture**: A simple LSTM-based model is used to capture sequential dependencies in the text.
   - **Training**: The model is trained on the `sampled_train` dataset, and training metrics are recorded.
   - **Model Saving**: After training, the model is saved as a PyTorch model artifact in the `/models` directory.

3. **Model Prediction**
   - **Prediction Script**: A separate script is used to load the saved model and perform predictions on new text inputs.
   - **API Development**: FastAPI is used to serve the model as a REST API.
   - **Endpoint Creation**: An endpoint `/predict` is created which accepts POST requests with text data and returns the prediction.

4. **Deployment**
   - **FastAPI Setup**: FastAPI is set up on localhost at port 8080.
   - **Testing**: The API is tested using curl commands to ensure it works as expected.
   
## Project Structure

- `train_model.py`: Script for training the model.
- `predict_model.py`: Script for loading the model and making predictions.
- `app.py`: FastAPI application script.
- `models/`: Directory to save the trained model.
- `results/`: Directory to save the test results.
- `requirements.txt`: File listing required Python packages.
- `project.md`: Documentation of the approach and thought process.

## API Usage

To call the API and get a prediction, you can use the following curl command:

```sh
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"text": "Your text here"}'
```
### Example Usage

1. **Start the FastAPI server**:
   ```sh
   uvicorn app:app --host 127.0.0.1 --port 8080
   ```
2. **Send a text input for prediction**:
   ```sh
   curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"text": "I hate this!"}'
   ```
   This will return the prediction result, indicating whether the text is classified as `hate` or `noHate`.

   Also you can use the following http://127.0.0.1:8080/docs

## Conclusion

This project offers solutions for text classification and can be adapted for various textual data analysis applications.

Primary goal is to construct a basic text classification model capable of differentiating between 'hate' and 'noHate' speech within a given dataset. 
