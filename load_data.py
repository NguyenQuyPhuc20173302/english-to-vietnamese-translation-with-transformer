import tensorflow as tf
import re
import string
from tensorflow.keras.layers import TextVectorization


def load_data(file_name):
    list_sentence = []
    with open(file_name, encoding='utf8') as f:
        for line in f.readlines():
            list_sentence.append(line)

    return list_sentence


def text_pairs(en, vn):
    text_pair = []
    for i in range(len(vn)):
        text_pair.append((en[i].split('\n')[0], "[start] " + vn[i].split('\n')[0] + " [end]"))

    return text_pair


vn_train = load_data('./data/train/data.vi')
en_train = load_data('./data/train/data.ja')

# vn_test = load_data('./data/test/test.tgt')
# en_test = load_data('./data/test/test.src')
#
# vn_val = load_data('./data/val/valid.tgt')
# en_val = load_data('./data/val/valid.src')

train_pairs = text_pairs(en_train, vn_train)
# test_pairs = text_pairs(en_test, vn_test)
# val_pairs = text_pairs(en_val, vn_val)

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 50
batch_size = 64


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="tf_idf"
)
spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="tf-idf",
    standardize=custom_standardization,
)
train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)


def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return {"encoder_inputs": eng, "decoder_inputs": spa[:, :-1], }, spa[:, 1:]


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
# val_ds = make_dataset(val_pairs)
