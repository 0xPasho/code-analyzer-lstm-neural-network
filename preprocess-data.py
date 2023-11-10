import json
import pickle
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# Preprocessing: Load and process the JSONL files to generate tokens and tokenizer
input_data_dir = 'test_data'
output_data_dir = 'test_data_prepared'
os.makedirs(output_data_dir, exist_ok=True)

code_token_strings = []
comment_token_strings = []

# Load and process the files
# for i in range(0, 14):
for i in range(0, 2):
    file_name = os.path.join(input_data_dir, f"python_train_{i}.jsonl")
    with open(file_name, 'r') as file:
        for line in file:
            data = json.loads(line)
            code_token_strings.append(' '.join(data['code_tokens']))
            comment_token_strings.append(
                ' '.join(data['docstring_tokens']))

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(code_token_strings + comment_token_strings)
code_tokens = tokenizer.texts_to_sequences(code_token_strings)
comment_tokens = tokenizer.texts_to_sequences(comment_token_strings)
print("All good, data has been preprocessed!")

# Prepare the sequences
max_code_length = max(len(seq) for seq in code_tokens)
max_comment_length = max(len(seq) for seq in comment_tokens)
max_sequence_length = max(max_code_length, max_comment_length)

padded_code_tokens = pad_sequences(
    code_tokens, maxlen=max_sequence_length, padding='post')
padded_comment_tokens = pad_sequences(
    comment_tokens, maxlen=max_sequence_length, padding='post')
comment_categories = [to_categorical(seq, num_classes=len(
    tokenizer.word_index)+1) for seq in padded_comment_tokens]

# Convert the padded sequences to a categorical format
num_classes = len(tokenizer.word_index) + 1
y_train = np.array([to_categorical(seq, num_classes=num_classes)
                   for seq in padded_comment_tokens])


# Convert to numpy arrays
X_train = np.array(padded_code_tokens)
y_train = np.array([seq[:max_sequence_length] for seq in y_train])

# Define the model
# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
model = Sequential()
model.add(Embedding(input_dim=vocab_size,
          output_dim=100, input_length=max_code_length))
model.add(LSTM(100, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)

# Save the trained model
model.save('my_model.h5')


def preprocess_and_predict(test_code_strings):
    # Tokenize and pad sequences
    test_code_tokens = tokenizer.texts_to_sequences(test_code_strings)
    padded_test_code_tokens = pad_sequences(
        test_code_tokens, maxlen=max_code_length, padding='post')

    # Predict
    predictions = model.predict(padded_test_code_tokens)
    return predictions


def predictions_to_text(predictions, tokenizer):
    reverse_tokenizer = {index: word for word,
                         index in tokenizer.word_index.items()}
    text_predictions = []
    for prediction in predictions:
        sequence = np.argmax(prediction, axis=-1)
        text = ' '.join(reverse_tokenizer.get(token, '?')
                        for token in sequence)
        text_predictions.append(text)
    return text_predictions


test_code_strings = ["def", "square",
                     "(", "num", ")", ":", "return", "num", "*", "num"]

# Preprocess and predict
predictions = preprocess_and_predict(test_code_strings)

# Convert predictions to text
predicted_comments = predictions_to_text(predictions, tokenizer)

# Now `predicted_comments` contains the comment predictions for your test code
for i, (code, comment) in enumerate(zip(test_code_strings, predicted_comments)):
    print(f"Code: {code}\nPredicted Comment: {comment}\n")
