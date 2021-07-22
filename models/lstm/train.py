


from logging import Logger
import os
import numpy as np
import argparse

from olist_sentiment_analysis import dataset
from olist_sentiment_analysis.models import lstm 
from olist_sentiment_analysis.utils.tensorflow_extended import ExtendedTensorBoard

import tensorflow as tf

import mlflow
import mlflow.tensorflow
mlflow.tensorflow.autolog()

from utils.logger import get_logger
from olist_sentiment_analysis.models.lstm.train import LOG

LOG: Logger = get_logger('trainer')

CHECKPOINT_PATH = os.path.dirname("training_1/cp.ckpt")
COLUMN_TEXT = 'text'
COLUMN_LABEL = 'text'

DEFAULT_DATASET_PATH = 'https://raw.githubusercontent.com/MarcosMota/AnaliseDeSentimento/master/dataset/olist_order_reviews_dataset.csv'

parser = argparse.ArgumentParser()
parser.add_argument("--path_dataset",default=DEFAULT_DATASET_PATH, type=int,help="dataset path")
parser.add_argument("--lr", default=0.001, type=int, help="learning rate")
parser.add_argument("--train_split", default= 0.8, type=int,help="number of divide dataset train")
parser.add_argument("--random_state", default= 42, type=int,help="random state")
parser.add_argument("--vocab_size", default= 10000, type=int,help="vocabulary size")
parser.add_argument("--embedding_dim", default= 16, type=int,help="number of embeddind dimension")
parser.add_argument("--max_length", default= 120, type=int,help="max word length in sentence")
parser.add_argument("--batch_size", default=128, type=int,help="batch size")
parser.add_argument("--num_epochs", default=5, type=int,help="number of training steps")
parser.add_argument("--early_stopping_criteria", default=2, type=int,help="early stop criteria")
parser.add_argument("--dropout", default=0.3, type=int,help="dropout percentage")
parser.add_argument("--model_storage", default="model_storage/lstm", type=int,help="model_storange")

def main(argv):
    with mlflow.start_run():
        args = parser.parse_args(argv[1:])

        clear_fn = dataset.clear_data_fn(column_text=COLUMN_TEXT, column_label=COLUMN_LABEL)
        df = dataset.load_from_file(args.path_dataset, clear_data_fn = clear_fn)
        LOG.info(f'Dataset loaded from {args.path_dataset} with {len(df)} observations.')

        tokenizer = dataset.fit_tokenizer(np.array(df['text'].values),args.vocab_size)
        LOG.info(f'Fit tokenization and creating vocabulary')

        transform_sequence = dataset.transform_to_sequence_fn(tokenizer=tokenizer, max_length=args.max_length)

        split_ds = dataset.split_ds_fn()

        train_dataset, val_dataset, test_dataset = dataset.preprocess_fn(df = df,
                                                                transfom_to_seq_fn = transform_sequence,
                                                                split_ds_fn = split_ds)
        LOG.info(f'Apply preprocess into dataset')

        model:tf.keras.Model = lstm.build(args)
        LOG.info(f'Build neural network lstm with {model.count_params()} parameters.')


        earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.early_stopping_criteria)
        tensorflowBoardCallback = ExtendedTensorBoard(test_dataset=test_dataset)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)

        callbacks = [earlyStoppingCallback, tensorflowBoardCallback, cp_callback]

        model.compile(
            loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer= tf.keras.optimizers.Adam(
                learning_rate=0.0001),
            metrics=['accuracy']
        )

        LOG.info(f'Compile model with lr: {args.lr}')

        history = model.fit(
            train_dataset,
            validation_set=val_dataset,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            callbacks= callbacks,
        )
        LOG.info(f'Fit model for {args.num_epochs}')

        # Evaluate the model
        print("Evaluate")
        result = model.evaluate(test_dataset)
        print(dict(zip(model.metrics_names, result)))


if __name__ == "__main__" :
    main(os.system.argv)
