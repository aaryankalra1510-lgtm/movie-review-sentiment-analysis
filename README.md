# 🎬 Movie Review Sentiment Analysis

A machine learning project that uses Natural Language Processing (NLP) to classify movie reviews as **positive** or **negative**.

The project uses TF-IDF to convert text into numerical features and Logistic Regression to predict the sentiment of a review.

## 📌 Project Overview

The goal of this project is to build a simple text-classification model that can analyze movie reviews and predict whether the sentiment expressed is positive or negative.

### Features

- Processes text using TF-IDF vectorization.
- Uses Logistic Regression for sentiment classification.
- Splits data into training and testing sets.
- Evaluates model performance using accuracy, precision, recall, and F1-score.
- Allows users to enter their own reviews for sentiment predictions.
- Saves the trained model for later use.

## 📊 Dataset

**Dataset:** Rotten Tomatoes Movie Reviews

**Source:** [Hugging Face — Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)

The dataset contains movie reviews labelled by sentiment.

| Label | Meaning |
|---|---|
| `0` | Negative |
| `1` | Positive |

For the version used in this project, the original dataset splits were combined and a new 80/20 training/testing split was created.

- **80%** — Training data
- **20%** — Testing data

A fixed random seed was used to make the split reproducible.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Google Colab

## 🧠 How the Model Works

1. **Data preparation:** Load the labelled movie reviews from CSV files.
2. **Text vectorization:** TF-IDF converts review text into numerical features.
3. **Model training:** Logistic Regression learns patterns associated with positive and negative sentiment.
4. **Evaluation:** The trained model predicts sentiment for the test reviews.
5. **Prediction:** New reviews can be entered to obtain sentiment predictions.
6. **Model saving:** Joblib saves the trained pipeline for future use.

## 📁 Project Structure

```text
movie-review-sentiment-analysis/
├── sentiment_model.joblib
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

- `sentiment_model.joblib` — Saved TF-IDF and Logistic Regression pipeline.
- `train_model.py` — Script for training the model.
- `predict.py` — Script for predicting sentiment on new reviews.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

*Note: Add the Python scripts to the repository if you have saved them as separate files.*

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
pandas
scikit-learn
joblib
```

## 🚀 Making Predictions

After installing the dependencies and obtaining the trained model, run:

```python
import joblib

model = joblib.load("sentiment_model.joblib")

review = ["The movie was fantastic and entertaining!"]
prediction = model.predict(review)[0]

if prediction == 1:
    print("Positive review")
else:
    print("Negative review")
```

Example output:

```text
Positive review
```

The example output is illustrative; actual predictions depend on the trained model.

## 📈 Model Evaluation

The model can be evaluated using:

- **Accuracy:** Percentage of test reviews classified correctly.
- **Precision:** How often predictions for a sentiment class are correct.
- **Recall:** How many reviews of a sentiment class the model identifies.
- **F1-score:** A combined measure of precision and recall.

The measured test results should be added here after running the evaluation code:

| Metric | Result |
|---|---|
| Test accuracy | Add your measured result |
| Precision | Add your measured result |
| Recall | Add your measured result |
| F1-score | Add your measured result |

## ⚠️ Limitations

- The model may struggle with sarcasm, irony, and complex language.
- Predictions depend on the quality and variety of the training data.
- The model is designed for movie-review sentiment and may not generalize well to other types of text.
- Predicted probabilities should not be interpreted as guaranteed confidence in the result.

## 🔮 Future Improvements

- Compare Logistic Regression with other classification algorithms.
- Experiment with different text-preprocessing techniques.
- Build a simple web interface for interactive predictions.
- Explore transformer-based models such as BERT.
- Test the model on additional review datasets.

## 📜 Dataset Attribution

The project uses the Rotten Tomatoes dataset available through Hugging Face. Please consult the dataset page for its attribution and usage terms.

## 👤 Author

Created as a beginner-friendly NLP and machine-learning project to explore text classification and model evaluation.
