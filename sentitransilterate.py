import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertTokenizerFast
import pandas as pd
# Define hyperparameters
max_length = 51
num_epochs = 3
batch_size = 32

# Load preprocessed data
data = pd.read_csv('hindi.csv')

# Split data into training and validation sets
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

# Tokenize text data using the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
train_encodings = tokenizer(train_data['text'].tolist(), max_length=max_length, padding=True, truncation=True)
test_encodings = tokenizer(test_data['text'].tolist(), max_length=max_length, padding=True, truncation=True)

# Convert encodings to TensorFlow tensors
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_data['label'].tolist()
)).shuffle(len(train_data)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_data['label'].tolist()
)).batch(batch_size)

# Load the BERT model
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

# Define the model architecture
input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
embedding_layer = bert_model(input_ids, attention_mask)[0]
x = GlobalAveragePooling1D()(embedding_layer)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_dataset)
print(f'Test loss: {loss:.2f}')
print(f'Test accuracy: {accuracy:.2f}')