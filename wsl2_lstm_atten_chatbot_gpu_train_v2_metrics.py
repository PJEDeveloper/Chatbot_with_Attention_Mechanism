import os
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle

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
BATCH_SIZE = 32
EPOCHS = 150  # Maximum number of epochs
MAX_LENGTH = 20
PATIENCE = 5  # Early stopping patience
RESULTS_DIR = '/mnt/d/Development/Projects/AI_Projects/Chatbot_w_Attention/model_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load TED Talks Translation Dataset
print("Loading TED Talks Translation Dataset...")
dataset_name = "ted_hrlr_translate/pt_to_en"
data, info = tfds.load(dataset_name, as_supervised=True, with_info=True)
train_data, val_data = data['train'], data['validation']

# Tokenize and preprocess dataset
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (sentence.numpy() for pair in train_data for sentence in pair),
    target_vocab_size=VOCAB_SIZE
)

START_TOKEN, END_TOKEN = [VOCAB_SIZE], [VOCAB_SIZE + 1]
tokenizer_vocab_size = VOCAB_SIZE + 2

def tokenize_sentence(sentence):
    return START_TOKEN + tokenizer.encode(sentence.numpy()) + END_TOKEN

def validate_tokens(tensor):
    return tf.clip_by_value(tensor, 0, tokenizer_vocab_size - 1)

def preprocess(inp, targ):
    inp = tf.py_function(tokenize_sentence, [inp], tf.int64)
    targ = tf.py_function(tokenize_sentence, [targ], tf.int64)
    inp = validate_tokens(inp)
    targ = validate_tokens(targ)
    inp.set_shape([None])
    targ.set_shape([None])
    return inp, targ

def filter_max_length(inp, targ):
    return tf.size(inp) > 1 and tf.size(targ) > 1 and tf.size(inp) <= MAX_LENGTH and tf.size(targ) <= MAX_LENGTH

train_dataset = (
    train_data.map(preprocess)
    .filter(filter_max_length)
    .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
    .cache('/tmp/train_cache')
    .repeat()
)

val_dataset = (
    val_data.map(preprocess)
    .filter(filter_max_length)
    .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
)

# Define Encoder, Decoder, and Attention classes
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

encoder = Encoder(tokenizer_vocab_size, LSTM_UNITS)
decoder = Decoder(tokenizer_vocab_size, LSTM_UNITS)

# Optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(input_batch, target_batch):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_state_h, enc_state_c = encoder(input_batch)
        dec_state_h, dec_state_c = enc_state_h, enc_state_c
        dec_input = tf.expand_dims([START_TOKEN[0]] * input_batch.shape[0], 1)

        for t in range(1, target_batch.shape[1]):
            predictions, dec_state_h, dec_state_c = decoder(dec_input, enc_output, dec_state_h, dec_state_c)
            loss += loss_object(target_batch[:, t], tf.squeeze(predictions, axis=1))
            dec_input = tf.expand_dims(target_batch[:, t], 1)

    gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return loss / int(target_batch.shape[1])

# Custom Training Loop with Validation Loss Calculation
num_batches = info.splits['train'].num_examples // BATCH_SIZE

for epoch in range(EPOCHS):
    total_loss = 0.0
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    for batch, (input_batch, target_batch) in enumerate(train_dataset.take(num_batches)):
        batch_loss = train_step(input_batch, target_batch)
        total_loss += batch_loss.numpy()
        if (batch + 1) % 10 == 0:
            print(f" Batch {batch + 1} Loss: {batch_loss.numpy():.4f}")

    avg_loss = total_loss / num_batches  # Use num_batches instead of batch + 1
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

# Build models for saving weights
encoder.build(input_shape=(None, MAX_LENGTH))
decoder.build(input_shape=[(None, 1), (None, MAX_LENGTH, LSTM_UNITS), (None, LSTM_UNITS), (None, LSTM_UNITS)])

# Save model weights
encoder.save_weights(os.path.join(RESULTS_DIR, 'encoder.weights.h5'))
decoder.save_weights(os.path.join(RESULTS_DIR, 'decoder.weights.h5'))

# Save tokenizer
tokenizer_path = os.path.join(RESULTS_DIR, 'tokenizer.pkl')
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print("Training complete. Model weights and tokenizer saved.")
