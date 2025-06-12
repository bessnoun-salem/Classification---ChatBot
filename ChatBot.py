# Import necessary libraries
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
from plotly.offline import plot
import plotly.graph_objs as go

from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Load the chatbot and resume datasets
chatbot = pd.read_csv("Sheet_1.csv", usecols=['response_id','class','response_text'], encoding='latin-1')
resume = pd.read_csv("Sheet_2.csv", encoding='latin-1')

# Display first few rows of both datasets (optional)
print(chatbot.head(5))
print(resume.head(5))

# Display distribution of classes in chatbot data
print(chatbot['class'].value_counts())

# Function to generate and display a word cloud of chatbot responses
def cloud(text):
    wordcloud = WordCloud(background_color="blue", stopwords=stop).generate(" ".join([i.upper() for i in text]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Chat Bot Response")
    plt.show()

cloud(chatbot['response_text'])  # Show word cloud for chatbot responses

# Encode the class labels to numeric values
chatbot['Label'] = Encode.fit_transform(chatbot['class'])

# Define features (texts) and labels
x = chatbot.response_text
y = chatbot.Label

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# Convert text data into numerical vectors using CountVectorizer
vect = CountVectorizer()
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

# Train a Multinomial Naive Bayes classifier
NB = MultinomialNB()
NB.fit(x_train_dtm, y_train)
y_predict = NB.predict(x_test_dtm)
print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_predict))

# Train a Random Forest classifier
rf = RandomForestClassifier(max_depth=10, max_features=10)
rf.fit(x_train_dtm, y_train)
rf_predict = rf.predict(x_test_dtm)
print("Random Forest Accuracy:", metrics.accuracy_score(y_test, rf_predict))

# Combine x_test and y_test with predictions into a single DataFrame for analysis
test_df = pd.DataFrame({
    'response_text': x_test.values,
    'true_label': y_test.values,
    'nb_predicted_label': y_predict,
    'rf_predicted_label': rf_predict
})

print("Sample of test data with predictions:")
print(test_df.head())

# (Optional) Save the test data with predictions to a CSV file
test_df.to_csv("test_data_with_predictions.csv", index=False, encoding='utf-8')

# Prepare text data for t-SNE visualization using CountVectorizer (max 256 features)
Chatbot_Text = chatbot["response_text"]
Tf_idf = CountVectorizer(max_features=256).fit_transform(Chatbot_Text.values)

# Apply t-SNE for dimensionality reduction to 3D
tsne = TSNE(
    n_components=3,
    init='random',         # Alternative: 'pca'
    random_state=101,
    method='barnes_hut',
    n_iter=300,            # Number of iterations, must be >= 250
    verbose=2,
    angle=0.5
).fit_transform(Tf_idf.toarray())

# Create a 3D scatter plot using Plotly to visualize t-SNE results
trace1 = go.Scatter3d(
    x=tsne[:, 0],
    y=tsne[:, 1],
    z=tsne[:, 2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color=chatbot['Label'],  # Color points by class label
        colorscale='Portland',
        colorbar=dict(title='Class Label'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

# Define figure layout
data = [trace1]
layout = dict(height=800, width=800, title='3D TSNE Visualization')
fig = dict(data=data, layout=layout)

# Show the plot in a browser
plot(fig, filename='3DBubble.html')
