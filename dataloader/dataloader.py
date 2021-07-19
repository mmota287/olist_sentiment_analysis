"""Data Loader"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
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

    @staticmethod
    def sentence_to_sequence(config, training_sentences, testing_sentences):
        """
        Transform sentence to sequence, after add padding

        Args:
            
            training_sentences ([array]): Training sentences
            testing_sentences ([array]): Test sentences

        Returns:
            training_padded : Training sequence with padding
            testing_padded: Test sequence with padding
        """
        tokenizer = DataLoader.loadTokenizer(config)
        tokenizer.fit_on_texts(training_sentences)

        training_sentences = tokenizer.texts_to_sequences(training_sentences)
        training_padded = pad_sequences(training_sentences,maxlen=config.max_length)

        testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sequences,maxlen=config.max_length)
        return training_padded, testing_padded

    @staticmethod
    def loadTokenizer(config):
        """ Load Tokeninzer
        Get vocabulay saved in disk, if notfound create new Tokenizer

        Args:
            config ([type]): [description]
        """
        oov_tok = "<OOV>"
        tokenizer = Tokenizer(num_words = config.vocab_size, oov_token=oov_tok)

    @staticmethod
    def split_dataset(dataset, config):
        """
        Split dataset

        Args:
            dataset ([type]): Pandas dataset
            config ([type]): Project configurations

        Returns:
            training_sentences: Training sentences
            training_labels: Test sentences
            training_labels: Training labels
            testing_labels: Test labels
        """
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

