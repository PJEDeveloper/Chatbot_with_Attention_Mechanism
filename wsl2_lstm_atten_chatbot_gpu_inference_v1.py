import os
import pickle
import tensorflow as tf

# Ensure TensorFlow uses GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU enabled.")
else:
    print("No GPU found, using CPU.")

# Constants
VOCAB_SIZE = 100000
LSTM_UNITS = 512
MAX_LENGTH = 20
RESULTS_DIR = '/mnt/d/Development/Projects/AI_Projects/Chatbot_w_Attention/model_results'

# Load tokenizer
tokenizer_path = os.path.join(RESULTS_DIR, 'tokenizer.pkl')
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded successfully.")

# Define Encoder and Decoder (reuse from training script)
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, units)
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CrossAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, units)
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = CrossAttention(units)

    def call(self, x, enc_output, state_h, state_c):
        context_vector, _ = self.attention(state_h, enc_output)
        x = tf.reshape(x, [-1, 1])
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x, initial_state=[state_h, state_c])
        x = self.fc(output)
        return x, state_h, state_c

# Initialize encoder and decoder
encoder = Encoder(VOCAB_SIZE + 2, LSTM_UNITS)  # +2 to include START/END tokens
decoder = Decoder(VOCAB_SIZE + 2, LSTM_UNITS)

# Build the models with dummy input to prepare for loading weights
encoder.build(input_shape=(None, MAX_LENGTH))
decoder.build(input_shape=[(None, 1), (None, MAX_LENGTH, LSTM_UNITS), (None, LSTM_UNITS), (None, LSTM_UNITS)])
print("Models built successfully.")

# Load model weights
encoder.load_weights(os.path.join(RESULTS_DIR, 'encoder.weights.h5'))
decoder.load_weights(os.path.join(RESULTS_DIR, 'decoder.weights.h5'))
print("Model weights loaded successfully.")

# Inference function
def generate_response(input_seq):
    enc_output, enc_state_h, enc_state_c = encoder(input_seq)
    dec_input = tf.expand_dims([VOCAB_SIZE], 0)  # <start> token
    result = []
    for _ in range(MAX_LENGTH):
        predictions, dec_state_h, dec_state_c = decoder(dec_input, enc_output, enc_state_h, enc_state_c)
        predicted_id = int(tf.argmax(predictions[0], axis=-1).numpy())
        result.append(predicted_id)
        if predicted_id == VOCAB_SIZE + 1:  # <end> token
            break
        dec_input = tf.expand_dims([predicted_id], 0)
    return result

def decode_response(response, tokenizer):
    return tokenizer.decode(response)

# Example usage
input_sentence = "Hello, how are you?"
input_seq = tokenizer.encode(input_sentence)  # Correctly encode the input sentence
input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=MAX_LENGTH, padding='post')
response = generate_response(tf.constant(input_seq))
decoded_response = decode_response(response, tokenizer)
print("Generated Response:", decoded_response)
