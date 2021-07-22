from olist_sentiment_analysis.dataset import SplitDatasetFnType, TransformToSeqFnType
from typing import Tuple
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from toolz import curry


@curry
def transform_to_sequence_fn(
        sentences: np.array,
        tokenizer: Tokenizer,
        max_length: int) -> np.array:
    """Transfom sentences into sequences padded

    Args:
        sentences (np.array): Sequences
        tokenizer (Tokenizer): Tensorflow Tokenizer
        max_length (int): [description]

    Returns:
        sequences_padded: Sequences padded
    """
    
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences_padded = pad_sequences(sequences, maxlen=max_length, truncating='post')
        
    return sequences_padded

@curry
def split_ds_fn(
        dataset: tf.data.Dataset,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15) -> Tuple[tf.data.Dataset, tf.data.Dataset,tf.data.Dataset]:
    """Split dataset

    Args:
        dataset (tf.data.Dataset): [description]

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset,tf.data.Dataset]: [description]
    """
    dataset_size = len(dataset)

    train_len = int(train_size * dataset_size)
    val_len = int(val_size * dataset_size)
    test_len = int(test_size * dataset_size)

    dataset = dataset.shuffle(buffer_size=12)
    train_dataset = dataset.take(train_len)
    test_dataset = dataset.skip(train_len)
    val_dataset = test_dataset.skip(val_len)
    test_dataset = test_dataset.take(test_len)

    return (train_dataset, val_dataset, test_dataset)


def preprocess_fn(
    df: pd.DataFrame, 
    transfom_to_seq_fn: TransformToSeqFnType, 
    split_ds_fn: SplitDatasetFnType) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ Process dataframe and split into training and testing

    Args:
        df (pd.DataFrame): Pandas dataframe
        transfom_to_seq_fn (TransformToSeqFnType): Transform function to sequence
        split_ds_fn (SplitDatasetFnType): Dataset division function
    Returns:
        [type]: [description]
    """
    sequences = transfom_to_seq_fn(np.array(df['text'].values))
    labels = np.array(df['label'].values)

    ds = tf.data.Dataset.from_tensor_slices((sequences, labels))
    train_dataset, val_dataset, test_dataset = split_ds_fn(ds)

    return (train_dataset, val_dataset, test_dataset)
