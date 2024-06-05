# Make a prediction on some input data
# Import necessary libraries
import re  # Regular expression operations for text processing
import pandas as pd
import pickle  # Object serialization and deserialization

# Load the pre-trained classifier model from a pickle file
model = pickle.load(open('models/classifier.pkl', 'rb'))

# Set options to display all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Define a function to classify input text
def getClassification(text):
    # Convert text to lowercase to standardize it
    text = text.lower()
    # Remove usernames, special characters, URLs, and "rt" for retweets
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    # Print the cleaned text as a pandas Series for debugging
    print(pd.Series([text]))
    # Predict the class of the cleaned text using the model and return the first result
    return model.predict(pd.Series([text]))[0]

def clean_text(text):
    # Convert text to lowercase to standardize it
    text = text.lower()
    # Remove usernames, special characters, URLs, and "rt" for retweets
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    return text

def make_predictions_from_csv(csv_file):
    # Read only the first 20 rows of the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file, nrows=20)
    # Clean the text data
    data['clean_text'] = data['text'].apply(clean_text)
    # Make predictions using the model
    data['predicted_label'] = model.predict(data['clean_text'])
    # Return the DataFrame with predicted labels
    return data

# Define a sample text, Example usage of the getClassification function
text = 'I dont hate you'
# Classify the text and print the result
getClassification(text)
print(model.predict(pd.Series([text]))[0])

# Example usage of the make_predictions_from_csv function
csv_file = 'data/all_files_dataset.csv'
predictions = make_predictions_from_csv(csv_file)

#Save the predictions to a csv file
predictions.to_csv('results/predictions.csv', index=False)

# Print predictions
print(predictions)

