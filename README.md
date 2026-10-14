# Movie Review Sentiment Analysis

## Overview
This project uses natural language processing (NLP) to classify movie reviews as positive or negative. It uses TF-IDF to convert text into numerical features and Logistic Regression for classification.

## Dataset
The project was trained and evaluated using labeled movie-review data. The text column contains each review, and the label indicates sentiment:
- `0`: Negative
- `1`: Positive

Record the exact dataset and split used for each experiment so results can be reproduced.

## Technologies
- Python
- pandas
- scikit-learn
- joblib

## Model Pipeline
1. Clean and prepare review text and labels.
2. Convert review text into TF-IDF features.
3. Train a Logistic Regression classifier.
4. Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix.

## Evaluation Results

The following results were recorded for one test run:

- **Test accuracy:** 75.81%
- **Negative reviews:** precision 0.76, recall 0.75, F1-score 0.76 (1,067 examples)
- **Positive reviews:** precision 0.75, recall 0.77, F1-score 0.76 (1,066 examples)
- **Macro average:** precision 0.76, recall 0.76, F1-score 0.76
- **Weighted average:** precision 0.76, recall 0.76, F1-score 0.76
- **Total test examples:** 2,133

### Confusion Matrix

Rows are actual labels and columns are predicted labels, in the order **Negative, Positive**:

|  | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 796 | 271 |
| Actual Positive | 245 | 821 |

This means 796 negative reviews and 821 positive reviews were classified correctly. The model misclassified 271 negative reviews as positive and 245 positive reviews as negative.

## Run Predictions

Place `predict.py` and `sentiment_model_improved.joblib` in the same directory, then install dependencies:

```bash
pip install -r requirements.txt
```

Example:

```bash
python predict.py "This movie was amazing and entertaining!"
```

The script prints the predicted sentiment and, if supported by the model, its estimated probability for that prediction. This probability is not a guarantee of correctness.

## Limitations
- The model may struggle with sarcasm, mixed opinions, slang, and context-dependent language.
- Performance can differ on reviews from a different dataset or writing style.
- Reported accuracy applies only to the particular test set used in this experiment.

## Future Improvements
- Compare TF-IDF settings with cross-validation.
- Evaluate on a separate dataset to test generalization.
- Compare with a pretrained transformer model such as DistilBERT.
- Inspect misclassified reviews to identify common errors.

## Attribution
Mention the dataset source and any tutorials, code, or AI tools used when publishing or submitting this project. Ensure the dataset and its license permit your intended use.
