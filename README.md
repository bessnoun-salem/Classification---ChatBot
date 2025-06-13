Text Classification with Model Comparison
🎯 Objective
Build a text classification system using two machine learning algorithms:

Naive Bayes (MultinomialNB)

Decision Tree
and compare their performance using accuracy, classification report, confusion matrix, and Bokeh visualizations.

🧹 1. Data Preprocessing
Dataset loaded from a CSV file Sheet_1.csv with columns: response_text (input), class (label).

Text cleaning steps:

Lowercasing

Removing digits

Removing punctuation

Removing English stopwords

Lemmatizing words (using WordNetLemmatizer from NLTK)

🧠 2. Feature Engineering
Used TF-IDF Vectorization to convert text into numeric vectors.

Selected bi-grams (1-2) and limited features to 1000 most frequent.

🔀 3. Splitting Dataset
Split the dataset into training and testing sets:

80% for training

20% for testing
(using train_test_split with random_state=42)

🤖 4. Model Training
Naive Bayes (MultinomialNB):

Trained on the TF-IDF features

Predictions made on test data

Decision Tree Classifier:

Used a for loop to train multiple models with different max_depth values: [3, 5, 7, 10, 15]

For each depth, calculated accuracy and saved the best-performing model

📊 5. Evaluation
Calculated:

Accuracy

Classification report (precision, recall, f1-score)

Confusion matrix

📈 6. Visualization (Bokeh)
Generated interactive visualizations:

Bar chart comparing model accuracies

Confusion matrix heatmaps for:

Naive Bayes

Best Decision Tree model

Line plot showing how Decision Tree accuracy changes with max_depth

All visualizations were saved to an HTML file:
📁 results.html

✅ Key Findings
Naive Bayes performed well with limited data and was efficient.

Decision Tree performance varied based on max_depth. A shallow tree underfits, while a deep one may overfit.

The best Decision Tree depth was selected based on highest accuracy.

