from transformers import pipeline

def predict_masked_word(sentence):
    # Load BERT for masked word prediction
    # Here we use "bert-base-uncased" which is pre-trained on lowercased English text.
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")

    # Get predictions for the masked word
    predictions = fill_mask(sentence)

    # Return predictions
    return predictions

if __name__ == "__main__":
    # Get sentence input from the user
    # The sentence must contain the special token [MASK]
    sentence = input("Enter a sentence with a [MASK] token (e.g., 'We turned down a narrow lane and passed through a small [MASK].'): ")

    # Predict the masked word
    predictions = predict_masked_word(sentence)

    # Display the predictions
    print("\nPredictions for the masked word:")
    for pred in predictions:
        # 'token_str' holds the predicted word and 'score' is the confidence
        print(f"Prediction: {pred['token_str'].strip()}, Score: {pred['score']:.4f}")
