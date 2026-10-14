import argparse
import joblib


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a movie review is positive or negative."
    )
    parser.add_argument(
        "review",
        nargs="?",
        help='Movie review text, for example: "This movie was wonderful!"',
    )
    args = parser.parse_args()

    review = args.review
    if not review:
        review = input("Enter a movie review: ").strip()

    if not review:
        print("Please enter some review text.")
        return

    # Keep this file in the same folder as the saved model.
    model_path = "sentiment_model_improved.joblib"

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Could not find {model_path}. Put the model file beside predict.py.")
        return

    prediction = int(model.predict([review])[0])
    label_names = {0: "Negative", 1: "Positive"}
    label = label_names.get(prediction, f"Label {prediction}")
    print(f"Prediction: {label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100
        print(f"Model confidence: {confidence:.1f}%")
        print("Note: this is the model's estimated probability, not a guarantee.")


if __name__ == "__main__":
    main()
