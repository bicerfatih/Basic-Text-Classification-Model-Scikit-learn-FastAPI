# Create a model and store the result as an artifact
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample
import pickle
import numpy as np
import re
from nltk.corpus import stopwords

# Constants
REPORT_DIR = './results'
TEXT_PATH = 'data/all_files/'
ANNOTATIONS_PATH = 'data/annotations_metadata.csv'
# Dictionary for expanding contractions into their full forms
CONTRACTION_MAPPING = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
    "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"
}

# Data preparation

# Set display options to show all columns and rows when printing DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Retrieve all text files from the specified path
files_text = glob.glob(TEXT_PATH + '*.txt')

# Create an empty list to store file paths
path_list = []
for file in files_text:
    path_list.append(file)  # Add each file path to the list

# Print the first two file paths for verification
print(path_list[:2])

# Create a DataFrame from the list of file paths
file_path = pd.DataFrame(path_list, columns=['file_name'])

# Extract file identifiers from the file names and store them in a new column
file_path['file_id'] = file_path['file_name'].str.split('/').str[2].str.split('.').str[0]

# Load text files and store their first lines
texts = []
for file in files_text:
    # Open each file, read the first line, and append it to the texts list
    texts.append(open(file, 'r', encoding="utf-8").readline())

# Keep only the 'file_id' column from the file_path DataFrame
file_path = file_path['file_id']
# Concatenate file IDs and texts into a single DataFrame
hate_speech = pd.concat([file_path, pd.DataFrame(texts, columns=['text'])], axis=1)
# Load the annotations metadata and drop unnecessary columns
annotations = pd.read_csv(ANNOTATIONS_PATH)
annotations.drop(['subforum_id', 'num_contexts'], axis=1, inplace=True)
print(annotations.head())

# Merge the annotations DataFrame with the hate_speech DataFrame on 'file_id'
hate_speech_df = pd.merge(annotations, hate_speech, how='inner', on='file_id')
# Filter the merged DataFrame for entries labeled as 'noHate' or 'hate'
hate_speech_df = hate_speech_df.loc[(hate_speech_df.label == 'noHate') | (hate_speech_df.label == 'hate')]
# Keep only the 'label' and 'text' columns
hate_speech_df = hate_speech_df[['label', 'text']]
# Print the shape of the final DataFrame, unique labels, and the first few rows
print(hate_speech_df.shape)
print(hate_speech_df.label.unique(), hate_speech_df.shape)
print(hate_speech_df.head())

# Save the final DataFrame to a CSV file
hate_speech_df.to_csv('data/all_files_dataset.csv', index=False)

# Data Preprocessing

Stopwords = set(stopwords.words('english'))
# Create a copy of the original text column in the DataFrame
hate_speech_df['text_orginal'] = hate_speech_df['text'].copy()
# Create an empty list to store processed sentences
sentences = []
for sentence in hate_speech_df['text']:
    # Define regex patterns for replacing spaces, URLs, mentions, and hashtags
    space_pattern = '\s+'
    giant_url_regex = 'http[s]?://...'
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'

    # Apply regex substitutions to clean up the sentence
    sentence = re.sub(space_pattern, ' ', sentence)
    sentence = re.sub(giant_url_regex, 'URLHERE', sentence)
    sentence = re.sub(mention_regex, 'MENTIONHERE', sentence)
    sentence = re.sub(hashtag_regex, 'HASHTAGHERE', sentence)

    # Additional regex substitutions for punctuation and special characters
    sentence = re.sub(r"[^A-Za-z0-9^, !. \/'+=]", " ", sentence)
    sentence = re.sub(r"!", " ", sentence)
    sentence = re.sub(r"</s>", " ", sentence)
    sentence = re.sub(r",", " ", sentence)
    sentence = re.sub(r"\.", " ", sentence)
    sentence = re.sub(r"\/", " ", sentence)
    sentence = re.sub(r"\^", " ^ ", sentence)
    sentence = re.sub(r"\+", " + ", sentence)
    sentence = re.sub(r"\-", " - ", sentence)
    sentence = re.sub(r"\=", " = ", sentence)
    sentence = re.sub(r"'", " ", sentence)
    sentence = re.sub(r"(\d+)(k)", r"\g<1>000 ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r" e g", " eg ", sentence)
    sentence = re.sub(r" b g", " bg ", sentence)
    sentence = re.sub(r" u s ", " american ", sentence)
    sentence = re.sub(r"\0s", "0", sentence)
    sentence = re.sub(r" 9 11 ", "911", sentence)
    sentence = re.sub(r"e - mail", "email", sentence)
    sentence = re.sub(r"j k", "jk", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    sentence = re.sub(r'@[A-Za-z0-9]+', '', sentence)
    sentence = re.sub(r'(\w)\1{2,}', r'\1\1', sentence)
    sentence = re.sub(r'\w(\w)\1{2,}', '', sentence)

    # Convert the sentence to lowercase
    sentence = sentence.lower()
    # Replace contractions using the contraction_mapping
    sentence = ' '.join([CONTRACTION_MAPPING[t] if t in CONTRACTION_MAPPING else t for t in sentence.split(" ")])
    sentence = ' '.join([word for word in sentence.split() if word.isalpha()])
    # Remove stopwords and short words
    sentence = ' '.join([word for word in sentence.split() if word.lower() not in Stopwords and len(word) >= 3])
    # Append the cleaned and processed sentence to the list
    sentences.append(sentence.replace('.', '').replace(',', '').replace("'", ""))

# Update the DataFrame with the cleaned sentences
processed_hate_speech_df = hate_speech_df
# Assign sentences to the 'text' column of the DataFrame
processed_hate_speech_df['text'] = sentences
# Replace empty strings with np.nan for handling missing data
processed_hate_speech_df['text'] = hate_speech_df['text'].replace('', np.nan)
# Drop rows with NaN values
processed_hate_speech_df = hate_speech_df.dropna(axis=0)
# Append markers to the beginning and end of each sentence for model processing
processed_hate_speech_df.loc[:, 'text'] = processed_hate_speech_df['text'].apply(lambda x: '_START_ ' + x + ' _END_')

text =processed_hate_speech_df['text']
processed_hate_speech_df.shape, len(text)
# Print the number of entries in the processed data
print(len(text))

# Save the cleaned and processed data to a CSV file
processed_hate_speech_df.to_csv('data/processed_all_files_dataset.csv', index=False)

# Display some examples of labels and their corresponding text before and after processing
for i in range(5):
    print("Label:", hate_speech_df['label'][i])
    print("Text:", hate_speech_df['text'][i])
    print("\n")

for i in range(5):
    print("Label:", processed_hate_speech_df['label'][i])
    print("Text:", processed_hate_speech_df['text'][i])
    print("\n")


#Training

# Load the preprocessed data
processed_hate_speech_df = pd.read_csv('./data/processed_all_files_dataset.csv')
print(processed_hate_speech_df)

# Split the preprocessed data into training and test sets with a fixed random state for reproducibility.
Xy_train, Xy_test = train_test_split(processed_hate_speech_df, random_state=0)

# Display the count of each class in the training set to understand the distribution.
Xy_train['label'].value_counts()

# Increase the number of samples in the minority class to balance the training dataset.
upsampling = True
if upsampling:
    # Separate the classes.
    train_hate = Xy_train[Xy_train.label == "hate"]
    train_nohate = Xy_train[Xy_train.label == "noHate"]
    # Upsample the 'hate' class to have the same number of samples as 'noHate'.
    train_hate_upsampled = resample(train_hate,
                                    replace=True,  # Sample with replacement
                                    n_samples=len(train_nohate),  # Match number in majority class
                                    random_state=0)  # Reproducible results
    # Combine the upsampled class with the majority class.
    train_upsampled = pd.concat([train_hate_upsampled, train_nohate])
else:
    # Use the original dataset if not upsampling.
    train_upsampled = Xy_train

# Build a processing pipeline with a Stochastic Gradient Descent classifier.
pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),  # Convert text to a matrix of token counts
    ('tfidf', TfidfTransformer()),  # Transform counts to normalized tf or tf-idf values
    ('nb', SGDClassifier())  # The classifier
])

# Split the upsampled training and the test datasets into inputs (X) and outputs (y).
X_train, y_train = train_upsampled["text"], train_upsampled["label"]
X_test, y_test = Xy_test["text"], Xy_test["label"]

# Fit the model to the training data and predict the test data.
model = pipeline_sgd.fit(X_train, y_train)
y_predict = model.predict(X_test)
# Display accuracy and F1 score of the model on test data.
print("Accuracy:", model.score(X_test, y_test))
print("F1 score:", f1_score(y_test, y_predict, pos_label="noHate"))

# Save the trained model to a file for later use.
pickle_out = open("./models/classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

# Evaluate the model with a confusion matrix.
matrix = confusion_matrix(y_test, y_predict, labels=["hate", "noHate"])
tn, fp, fn, tp = matrix.ravel()

rates = {
        "False Positive Rate": fp / (fp + tn),
        "False Negative Rate": fn / (fn + tp),
        "True Positive Rate (Sensitivity)": tp / (tp + fn),
        "True Negative Rate (Specificity)": tn / (tn + fp)
    }

# Calculate and print rate statistics from the confusion matrix.
print(rates)
print("Confusion matrix:")
print(matrix)

# Save performance metrics, rates, and confusion matrix to the '/results' directory.
rates_df = pd.DataFrame([rates])
rates_df.to_csv(f'{REPORT_DIR}/training_report_rates.csv', index=False)
confusion_df = pd.DataFrame(matrix, index=['Actual Hate', 'Actual noHate'],
                                columns=['Predicted Hate', 'Predicted noHate'])
confusion_df.to_csv(f'{REPORT_DIR}/training_report_confusion_matrix.csv', index=True)

print("Saved performance metrics, rates, and confusion matrix to the '/results' directory.")
# The End

# This is Additional: Enhanced Features for Further Analysis
# Analysing the distribution of word counts.
# To maintain the directory structure, I added the code here instead of creating a distribution_sequences.py file.
import matplotlib.pyplot as plt
import pandas as pd

try:
    # Load the dataset
    hate_speech_df = pd.read_csv('./data/all_files_dataset.csv')

    # Ensure 'label' and 'text' columns are strings and handle missing values
    hate_speech_df['label'] = hate_speech_df['label'].fillna('').astype(str)
    hate_speech_df['text'] = hate_speech_df['text'].fillna('').astype(str)

    # Initialize lists to hold the word counts for labels and texts
    plot_label = []
    plot_text = []

    # Loop through the DataFrame and calculate word counts for both 'label' and 'text'
    for _, row in hate_speech_df.iterrows():
        plot_label.append(len(row['label'].split()))
        plot_text.append(len(row['text'].split()))

    # Create a DataFrame from the word count lists
    length_df = pd.DataFrame({'label': plot_label, 'text': plot_text})

    # Plot histograms
    fig, ax = plt.subplots(figsize=(16, 6))
    length_df.hist(bins=30, ax=ax)
    ax.set_title('Word Count Distribution in Labels and Texts')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')

    # Display the plot
    plt.show()

except FileNotFoundError:
    print("Error: The file 'all_files_dataset.csv' was not found.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: There was a problem parsing the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# This is Additional: Enhanced Features for Further Analysis
# Analysing text documents using FastText embeddings and computing Wasserstein distances.
# To maintain the directory structure, I added the code here instead of creating a compute_distribution.py file.

import numpy as np
from scipy import spatial
from tqdm import tqdm
import fasttext
import ot

try: # Error handling
    # Function to compute the frequency distribution of words in a document based on a given model's vocabulary.
    def compute_distribution(document, model):
        distr_A = []
        for word in model:
            # Check if each word in the model is in the document
            if word not in document.split(" "):
                distr_A.append((word, 0))
            else:
                cpt = 0
                # Count the occurrences of the word in the document
                for elt in document.split(" "):
                    if elt == word:
                        cpt += 1
                distr_A.append((word, cpt))
        # Calculate distribution, excluding words not in the model
        cpt = 0
        for elt in document.split(" "):
            if elt not in model:
                cpt += 1
        distribution = [x[1] / float(len(document.split(" ")) - cpt) for x in distr_A]
        return distribution

    # Function to compute the Wasserstein distance between two documents using a precise optimal transport algorithm.
    def Wasserstein_dist_OT(A, B, M, model):
        a = compute_distribution(A, model)
        b = compute_distribution(B, model)
        Wd = ot.emd2(a, b, M)  # Exact linear program for solving the OT problem
        return Wd

    # Function to compute the Wasserstein distance using the Sinkhorn algorithm for regularization.
    def Sinkhorn_dist_OT(A, B, M, reg, model):
        a = compute_distribution(A, model)
        b = compute_distribution(B, model)
        Wd = ot.sinkhorn(a, b, M, reg)  # Sinkhorn algorithm for entropy regularization
        return Wd

    # Function to export a DataFrame to an Excel file.
    def write_excel(df, out='df_out.xlsx', idx=False):
        writer = pd.ExcelWriter(out, engine='xlsxwriter', options={'strings_to_urls': False})
        df.to_excel(writer, index=idx)
        writer.save()
        writer.close()

    # Preparing data for FastText unsupervised training and initializing training.
    n = len(text) # size of training

    processed_hate_speech_df['text'].to_csv('results/corpus.txt', index=False, header=False) # csv for training fasttext
    print('Start FastText training...')

    model = fasttext.train_unsupervised("results/corpus.txt", model='cbow')
    print('FastText training done !')

    model.save_model("results/model.bin")
    # Creating and exporting embeddings for each word in the model.
    embedding = {}
    for w in model.words:
        embedding[w] = model[w]
    emb_df = pd.DataFrame.from_dict(embedding)
    transposed_df = emb_df.transpose()
    file_out = 'results/embeddings.xlsx'

    with pd.ExcelWriter(file_out, engine='xlsxwriter') as writer:
        transposed_df.to_excel(writer, sheet_name='Sheet1', index=True)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        text_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'font_name': 'Arial', 'font_size': 10})
        worksheet.set_column('A:Z', None, text_format)

    print(emb_df.head())

    # Words embedding matrix
    X = np.array([x.tolist() for x in embedding.values() if x.shape[0] != 1])
    # Coast matrix calculus (distances between words)
    print('Start coast matrix calculus')

    C_light = spatial.distance.pdist(X)
    dim = len(model.words)
    vals = C_light
    Z = np.zeros([dim,dim], dtype=np.double)
    Z[np.triu_indices(dim, k=1)] = vals
    Z = Z + Z.T
    Z[np.diag_indices(dim)] = 0
    distance = Z
    print('Coast matrix calculated')

    M = np.array(distance)
    text =processed_hate_speech_df['text']

    # Histograms of word distributions for each document, and computation of distances.
    try:
        docs_histogs = []
        for doc in tqdm(text, total=text.shape[0]):
            docs_histogs.append(compute_distribution(doc, model.words))
    except ZeroDivisionError:
        pass

    # Distance between 2 documents
    a = docs_histogs[0]
    b = docs_histogs[1]
    Wd = ot.emd2(a, b, M)

    # Computing a matrix of Wasserstein distances between a subset of documents.
    m = 10 # Small matrix size for testing
    res = []
    i = 0
    for x in processed_hate_speech_df.iloc[:m, 1]:
        for y in tqdm(processed_hate_speech_df.iloc[i:m, 1]):
            res.append(Wasserstein_dist_OT(x, y, M, model.words))
        i = i + 1

except Exception as e:
    print(f"An error occurred: {e}")