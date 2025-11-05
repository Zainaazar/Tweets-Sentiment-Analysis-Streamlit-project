def predict_sentiment(text):
    # Clean and tokenize (using the previously fitted tok object)
    seq = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen= max_len_seq, padding='post', truncating='post')

    # Predict (using the trained model_hybrid)
    pred = model_hybrid.predict(padded)
    label_index = np.argmax(pred, axis=1)[0]
    sentiment = label_map[label_index]

    return sentiment