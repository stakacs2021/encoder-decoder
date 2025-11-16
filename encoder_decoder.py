import sys
assert sys.version_info >= (3, 7)

# Warning: Using Keras 2 for this chapter
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras

from packaging import version
import tensorflow as tf
assert version.parse(tf.__version__) >= version.parse("2.8.0")

import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print("TensorFlow version:", tf.__version__)

# Check for GPU
if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. Neural nets can be very slow without a GPU.")
else:
    print("GPU detected:", tf.config.list_physical_devices('GPU'))

# Generate synthetic date dataset
# Input format: "Month Day, Year" (e.g., "April 22, 2019")
# Output format: "YYYY-MM-DD" (e.g., "2019-04-22")

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def generate_date_pairs(num_samples=100000, start_year=1900, end_year=2100):
    """Generate pairs of (input_date, output_date) where input is 'Month Day, Year' 
    and output is 'YYYY-MM-DD'"""
    pairs = []
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    for _ in range(num_samples):
        # Generate a random date
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        date = start_date + timedelta(days=random_days)
        
        # Format input: "Month Day, Year"
        month_name = MONTHS[date.month - 1]
        input_date = f"{month_name} {date.day}, {date.year}"
        
        # Format output: "YYYY-MM-DD"
        output_date = date.strftime("%Y-%m-%d")
        
        pairs.append((input_date, output_date))
    
    return pairs

print("\nGenerating date pairs...")
date_pairs = generate_date_pairs(num_samples=100000)
print(f"Generated {len(date_pairs)} date pairs")

# Show some examples
print("\nSample date pairs:")
for i in range(5):
    print(f"  {date_pairs[i][0]} => {date_pairs[i][1]}")

# Shuffle the data
random.shuffle(date_pairs)
sentences_in, sentences_out = zip(*date_pairs)

# Split into train, validation, and test sets
train_size = int(0.8 * len(sentences_in))
valid_size = int(0.1 * len(sentences_in))

train_in = sentences_in[:train_size]
valid_in = sentences_in[train_size:train_size + valid_size]
test_in = sentences_in[train_size + valid_size:]

train_out = sentences_out[:train_size]
valid_out = sentences_out[train_size:train_size + valid_size]
test_out = sentences_out[train_size + valid_size:]

print(f"\nDataset splits:")
print(f"  Train: {len(train_in)}")
print(f"  Validation: {len(valid_in)}")
print(f"  Test: {len(test_in)}")

# Set up text vectorization
vocab_size = 100  # Should be enough for digits, hyphens, months, etc.
max_length = 50  # Should be enough for date strings

text_vec_layer_in = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length, split="character"
)

text_vec_layer_out = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length, split="character"
)

# Adapt the vectorization layers
text_vec_layer_in.adapt(list(train_in))
text_vec_layer_out.adapt([f"startofseq {s} endofseq" for s in train_out])

print(f"\nInput vocabulary size: {text_vec_layer_in.vocabulary_size()}")
print(f"Output vocabulary size: {text_vec_layer_out.vocabulary_size()}")

# Prepare training data
X_train = tf.constant(list(train_in))
X_valid = tf.constant(list(valid_in))
X_test = tf.constant(list(test_in))

# Decoder inputs: add "startofseq" prefix
X_train_dec = tf.constant([f"startofseq {s}" for s in train_out])
X_valid_dec = tf.constant([f"startofseq {s}" for s in valid_out])
X_test_dec = tf.constant([f"startofseq {s}" for s in test_out])

# Decoder outputs: add "endofseq" suffix
Y_train = text_vec_layer_out([f"{s} endofseq" for s in train_out])
Y_valid = text_vec_layer_out([f"{s} endofseq" for s in valid_out])
Y_test = text_vec_layer_out([f"{s} endofseq" for s in test_out])

# Build the encoder-decoder model with attention
embed_size = 128

encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

# Vectorize inputs
encoder_input_ids = text_vec_layer_in(encoder_inputs)
decoder_input_ids = text_vec_layer_out(decoder_inputs)

# Embeddings
encoder_embedding_layer = tf.keras.layers.Embedding(
    text_vec_layer_in.vocabulary_size(), embed_size, mask_zero=True
)
decoder_embedding_layer = tf.keras.layers.Embedding(
    text_vec_layer_out.vocabulary_size(), embed_size, mask_zero=True
)

encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

# Encoder: Bidirectional LSTM
encoder = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
)
encoder_outputs, *encoder_state = encoder(encoder_embeddings)

# Concatenate forward and backward states
encoder_state = [
    tf.concat(encoder_state[::2], axis=-1),  # short-term (0 & 2)
    tf.concat(encoder_state[1::2], axis=-1)  # long-term (1 & 3)
]

# Decoder: LSTM
decoder = tf.keras.layers.LSTM(512, return_sequences=True)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

# Attention mechanism
attention_layer = tf.keras.layers.Attention()
attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

# Output layer
output_layer = tf.keras.layers.Dense(
    text_vec_layer_out.vocabulary_size(), activation="softmax"
)
Y_proba = output_layer(attention_outputs)

# Create the model
model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba])

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# Train the model
print("\nTraining the model...")
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "date_conversion_model",
    monitor="val_accuracy",
    save_best_only=True
)

history = model.fit(
    (X_train, X_train_dec),
    Y_train,
    epochs=10,
    validation_data=((X_valid, X_valid_dec), Y_valid),
    callbacks=[model_ckpt],
    batch_size=32
)

print("\nTraining completed!")

# Create inference model (without teacher forcing)
def translate_date(sentence_in):
    """Convert date from 'Month Day, Year' format to 'YYYY-MM-DD' format"""
    translation = ""
    for word_idx in range(max_length):
        X = np.array([sentence_in])  # encoder input
        X_dec = np.array(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec), verbose=0)[0, word_idx]
        predicted_word_id = np.argmax(y_proba)
        predicted_char = text_vec_layer_out.get_vocabulary()[predicted_word_id]
        
        if predicted_char == "endofseq":
            break
        translation += predicted_char
    
    return translation.strip()

# Test the model
print("\nTesting on some examples:")
test_examples = [
    "April 22, 2019",
    "December 31, 2000",
    "January 1, 2020",
    "March 15, 2023"
]

for example in test_examples:
    result = translate_date(example)
    print(f"  {example} => {result}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = []
for i, example in enumerate(test_in[:100]):  # Test on first 100 examples
    predicted = translate_date(example)
    actual = test_out[i]
    test_results.append(predicted == actual)

accuracy = sum(test_results) / len(test_results)
print(f"Test accuracy (exact match): {accuracy:.2%}")

