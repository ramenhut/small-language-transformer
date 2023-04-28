
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os as os
import random
import re

# The folder to load text files from.
path = 'shakes'
# Back propagation through time.
bptt = 100
# Number of attention heads.
num_heads = 8
# Dimension of the embeddings.
embed_dim = 32
# Dimension of the transformer ff layers.
ff_dim = 32
# Dropout rate.
dropout_rate = 0.1
# Number of most frequent tokens to use.
max_tokens = 30000
# Size of a batch used for training.
batch_size = 100
# Number of epochs used for training.
epoch_count = 10000
# Number of attention layers.
attention_layer_count = 6
# Tokenize by word or character?
tokenize_by_char = True
# Maximum number of batches per epoch. Useful if you 
# want to inspect the quality of prediction on a 
# set interval.
max_steps_per_epoch = 1000
# Maximum number of files to use.
max_files = 0

files_at_path = os.listdir(path)
files = []

for index in range(len(files_at_path)):
  if max_files != 0 and index == max_files:
    break
  filename = files_at_path[index]
  print("Parsing file: " + path + '/' + filename)
  contents = open(path + '/' + filename, 'r').read()
  files.append(contents)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_tokens, lower=False, char_level=tokenize_by_char)
tokenizer.fit_on_texts(files)
sequences = tokenizer.texts_to_sequences(files)
word_index = tokenizer.word_index

# Simplify our word index and index word to the max_tokens
word_index = { w:i for w,i in word_index.items() if i < max_tokens }
index_word = { i:w for w,i in word_index.items() }

print('Found %s unique tokens.' % len(word_index))

batch_count = 0
for sequence in sequences:
  batch_count += (len(sequence) - 1) // bptt

print("Creating " + str(batch_count) + " batches of size " + str(bptt) + " each.")

def generator(batch_size=128):
  # Our first axis iterates our batches, second axis is our 
  # time series (sequence of words in a sentence), and our
  # third axis is our one-hot encoding for each word.
  v_samples = np.zeros((batch_size, bptt))
  v_targets = np.zeros((batch_size, ))

  while 1:
    batch_index, token_index = 0, 0
    for sequence in sequences:
      for offset in range(bptt):
        for src_index in range((len(sequence) - offset - 1) // bptt * bptt):
          v_samples[batch_index, token_index] = sequence[src_index + offset] - 1
          token_index += 1
          if token_index == bptt:
            # We capture one 'next word' for each batch.
            v_targets[batch_index] = sequence[src_index + offset + 1] - 1
            token_index = 0
            batch_index += 1
          if batch_index == batch_size:
            yield v_samples, v_targets
            batch_index = 0
            v_samples = np.zeros((batch_size, bptt))
            v_targets = np.zeros((batch_size, ))

ffn = None
input = tf.keras.Input(shape=(bptt, ))
positions = tf.range(start=0, limit=bptt, delta=1)
token_embed = tf.keras.layers.Embedding(len(word_index), embed_dim)(input)
pos_embed = tf.keras.layers.Embedding(bptt, embed_dim)(positions)
input_embed = token_embed + pos_embed

for i in range(attention_layer_count):
    if i == 0:
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=(embed_dim // num_heads))(input_embed, input_embed)
    else:
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=(embed_dim // num_heads))(ffn, ffn)
    attention = tf.keras.layers.Dropout(dropout_rate)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_embed + attention)

    ffn = tf.keras.layers.Dense(ff_dim, activation='relu')(attention)
    ffn = tf.keras.layers.Dense(embed_dim)(ffn)
    ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
    ffn = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + ffn)
    
output = tf.keras.layers.GlobalAveragePooling1D()(ffn)
output = tf.keras.layers.Dropout(dropout_rate)(output)
output = tf.keras.layers.Dense(20, activation='relu')(output)
output = tf.keras.layers.Dropout(dropout_rate)(output)

output = tf.keras.layers.Dense(len(word_index), activation='softmax')(output),
model = tf.keras.Model(input, output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
model.summary()

data_generator = generator(batch_size=batch_size)

for epoch in range(epoch_count):
  model.fit(data_generator, steps_per_epoch=(batch_count if max_steps_per_epoch == 0 else max_steps_per_epoch), epochs=1)

  (samples, targets) = next(data_generator)
  target_index = random.randint(0, samples.shape[0] - 1)
  input = samples[target_index:target_index + 1, :]
  output_string = ''

  for i in range(100):
    output = model.predict(input, verbose=0)
    output_one_hot = np.zeros_like(output[0])
    max_index = np.random.choice(range(len(word_index)), p=output[0].ravel())
    output_one_hot[0, max_index] = 1
    output_string = output_string + index_word[max_index + 1] + ('' if tokenize_by_char else ' ')
    input = np.concatenate((input, np.array([[max_index]])), axis=1)
    input = input[:, 1:]

  print(output_string)
