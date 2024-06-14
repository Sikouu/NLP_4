import os
import time
import tensorflow as tf
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re

start_time = time.time()

data_dir = './jyxstxtqj_downcc.com'
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

texts = []
for file in files:
    with open(file, encoding='ansi', errors='ignore') as f:
        texts.append(f.read())

dataset = Dataset.from_dict({"text": texts})

class CustomTokenizer:
    def __init__(self, texts):
        self.vocab = self.build_vocab(texts)
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(re.findall(r'\w+|\S', text))
        return sorted(tokens)

    def encode(self, text):
        return [self.token_to_id[token] for token in re.findall(r'\w+|\S', text)]

    def decode(self, token_ids):
        return ''.join([self.id_to_token[token_id] for token_id in token_ids])

tokenizer = CustomTokenizer(texts)

def tokenize_function(examples):
    return {"input_ids": [tokenizer.encode(text) for text in examples["text"]]}

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

max_length = 512
input_ids = tf.keras.preprocessing.sequence.pad_sequences(
    [x for x in tokenized_datasets['input_ids']], maxlen=max_length, padding='post'
)

train_input_ids, val_input_ids = train_test_split(input_ids, test_size=0.1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_input_ids, train_input_ids)).shuffle(len(train_input_ids)).batch(4)
val_dataset = tf.data.Dataset.from_tensor_slices((val_input_ids, val_input_ids)).batch(4)

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_length=512):
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = self.add_weight(name="positional_encoding", shape=(1, max_seq_length, d_model), initializer='zeros')
        self.encoder_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model//nhead)
            for _ in range(num_layers)
        ]
        self.decoder_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model//nhead)
            for _ in range(num_layers)
        ]
        self.fc_out = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False, tgt=None):
        src = self.embedding(inputs[0]) + self.positional_encoding[:, :inputs[0].shape[1], :]
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, memory, return_attention_scores=False, training=training)
        output = tgt
        if output is None:
            return None  
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory, return_attention_scores=False, training=training)
        return self.fc_out(output)



vocab_size = tokenizer.vocab_size
model = TransformerModel(vocab_size)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

with tf.device(device):
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in: {elapsed_time // 3600}h {(elapsed_time % 3600) // 60}m {elapsed_time % 60}s")

model.save_weights("./transformer-finetuned-novels-tf.h5")
np.save("./transformer-finetuned-novels-tf-vocab.npy", tokenizer.vocab)

# 生成文本
def generate_text(model, tokenizer, prompt, max_length=200):
    input_ids = tokenizer.encode(prompt)
    output_ids = input_ids
    for _ in range(max_length):
        output = model(tf.constant([output_ids]), tf.constant([output_ids]))
        next_token_id = tf.argmax(output[0, -1, :])
        output_ids.append(next_token_id.numpy())
        if next_token_id == tokenizer.token_to_id.get('[SEP]', -1):
            break
    return tokenizer.decode(output_ids)

# 输入小说片段
prompt = "令狐冲淡然一笑，道：令狐冲死在姑娘的言语之下，那也不错啊。"

generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)
