
# Project Documentation

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

## File Descriptions

### app.py

**Purpose**:
- To serve the trained text classification model as a REST API using FastAPI.

**Functionality**:
- Imports necessary libraries including FastAPI, Pydantic, and PyTorch.
- Defines a FastAPI application.
- Loads the trained model and tokenizer.
- Defines a prediction function.
- Creates a POST endpoint `/predict` that accepts text input and returns the prediction.

### train_model.py

**Purpose**:
- To train a text classification model using a hate speech dataset.

**Functionality**:
- **Imports and Constants**:
  - Imports libraries such as `pandas`, `nltk`, `scikit-learn`, `pickle`, and `numpy` for data processing, model training, and evaluation.
  - Defines constants for report directory, data paths, and a dictionary for expanding contractions.

- **Data Preprocessing**:
  - **Text Cleaning**: Implements functions for cleaning text data, such as `expand_contractions`, `remove_stopwords`, and `clean_text`.
    - `expand_contractions(text, contraction_mapping)`: Expands contractions in text.
    - `remove_stopwords(text)`: Removes stopwords from text.
    - `clean_text(text)`: Performs text cleaning by lowercasing, removing special characters, etc.
  - **Data Loading**: Reads the dataset from CSV files using pandas.
  - **Balancing Dataset**: Uses resampling to handle class imbalance by upsampling the minority class.

- **Feature Extraction**:
  - Vectorizes the text data using `TfidfVectorizer` from `sklearn`.

- **Model Training**:
  - Trains a logistic regression model using the vectorized text data.
  - Uses cross-validation to evaluate model performance.
  - Saves the trained model using `pickle`.

- **Evaluation**:
  - Evaluates the model on the test dataset.
  - Generates classification reports and confusion matrices.
  - Saves results to the `results` directory.
  
  ```Additional Features for Further Analysis```
- **Embedding and Distance Calculation**:
  - Uses word embeddings to calculate distances between words.
  - Calculates Wasserstein distances between documents using the Optimal Transport method.
  - Generates word embeddings using pre-trained models and saves the results to an Excel file.
  
### `predict_model.py`

**Purpose**:
- To load the trained model and perform predictions on new text inputs.

**Functionality**:
- Imports necessary libraries including PyTorch and torchtext.
- Loads the saved model and tokenizer.
- Defines a function to preprocess and predict the class of new text inputs.

### `README.md`

**Purpose**:
- To provide instructions and details about the project, including goals, requirements, and submission guidelines.

**Functionality**:
- Explains the project goal and requirements.
- Provides instructions on the format and structure of the project.
- Details the requirements for the model, API, and documentation.
- Includes submission guidelines.

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

My primary goal is to construct a basic text classification model capable of differentiating between 'hate' and 'noHate' speech within a given dataset. The challenge emphasizes not just the performance of the model but rather the approach and thought process behind its development.

Using FastAPI, I have successfully deployed the model as a REST API, demonstrating a practical implementation of machine learning models in a real-world application. The API is designed to be interacted with via CURL commands, allowing users to easily test the model with text inputs and receive predictions.

Throughout the development process, I adhered to the specified requirements, maintained the directory structure as outlined (including few extra files). All additional dependencies have been listed in the `requirements.txt` file, ensuring reproducibility. Model training was performed on the provided dataset, with the model artifact saved under `/models` and test results documented under `/results`.

This project showcases my capability in handling sensitive NLP tasks with a clear focus on thoughtful data processing, model training, and careful deployment. 

The documentation within `./project.md` also provides insights into my decision-making and strategic planning. 

