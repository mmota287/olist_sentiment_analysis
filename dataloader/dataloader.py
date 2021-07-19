"""Data Loader"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(config):
        """Loads dataset from path"""
        df = pd.read_csv(config.path)
        df = df.dropna(subset=[config.column_text])

        df['label'] = pd.cut(df[config.column_score], bins=config.points_cut, labels=config.classes)
        df = df.rename(columns={config.column_text: 'text'})
        df = df[['text','label']]
        return df

    # @staticmethod
    # def validate_schema(data_point):
    #     jsonschema.validate({'image': data_point.tolist()}, SCHEMA)

    @staticmethod
    def preprocess_data(dataset, config):
        """ Preprocess and splits into training and test"""

        training_sentences, training_labels, testing_sentences, testing_labels = DataLoader.split_dataset(dataset, config)

        training_sentences = DataLoader.sentence_to_sequence(config, training_sentences, testing_sentences)

        return training_sentences, training_labels, testing_sentences, testing_labels

        train = dataset['train'].map(lambda image: DataLoader._preprocess_train(image, image_size),
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = dataset['test'].map(lambda image: DataLoader._preprocess_test(image, image_size))

        train_dataset = train.shuffle(buffer_size).batch(batch_size).cache().repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_dataset = test.batch(batch_size)

        return train_dataset, test_dataset

    @staticmethod
    def sentence_to_sequence(config, training_sentences, testing_sentences):
        oov_tok = "<OOV>"
        tokenizer = Tokenizer(num_words = config.vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(training_sentences)

        training_sentences = tokenizer.texts_to_sequences(training_sentences)
        training_padded = pad_sequences(training_sentences,maxlen=max_length, truncating=trunc_type)

        testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
        return training_sentences

    @staticmethod
    def split_dataset(dataset, config):
        df_train = dataset.sample(frac = config.split_train, random_state = config.random_state)
        df_test = dataset.drop(df_train.index)
        
        training_sentences = []
        training_labels = []
        testing_sentences = []
        testing_labels = []
        for index, train in df_train.iterrows():
            training_sentences.append(str(train['text']))
            training_labels.append(train['label'])
            
        for index, test in df_test.iterrows():
            testing_sentences.append(str(test['text']))
            testing_labels.append(test['label'])
            
        return training_sentences, training_labels, np.array(training_labels), np.array(testing_labels)
    
    @staticmethod
    def _normalize(input_image, input_mask):
        """ Normalise input image
        Args:
            input_image (tf.image): The input image
            input_mask (int): The image mask
        Returns:
            input_image (tf.image): The normalized input image
            input_mask (int): The new image mask
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

