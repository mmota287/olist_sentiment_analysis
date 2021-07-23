from logging import Logger
import os
import numpy as np
import argparse

from dataset.load import load_from_file, clear_data_fn
from dataset.preprocess import transform_to_sequence_fn, split_ds_fn, preprocess_fn
from dataset.vectorizer import fit_tokenizer

from models.lstm import model as lstm
from utils.tensorflow_extended import ExtendedTensorBoard

import tensorflow as tf

import mlflow
import mlflow.tensorflow
mlflow.tensorflow.autolog()

from utils.logger import get_logger

LOG = get_logger('LSTM_Trainer')

CHECKPOINT_PATH = os.path.dirname("checkpoints/lstm/cp.ckpt")
MODEL_LOG_PATH = 'model_log/lstm'
COLUMN_TEXT = 'review_comment_message'
COLUMN_LABEL = 'review_score'

DEFAULT_DATASET_PATH = 'https://raw.githubusercontent.com/MarcosMota/AnaliseDeSentimento/master/dataset/olist_order_reviews_dataset.csv'



parser = argparse.ArgumentParser()
parser.add_argument("--path_dataset",default=DEFAULT_DATASET_PATH, type=str, help="dataset path")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--train_split", default=0.8, type=float,help="number of divide dataset train")
parser.add_argument("--random_state", default= 42, type=int,help="random state")
parser.add_argument("--vocab_size", default= 10000, type=int,help="vocabulary size")
parser.add_argument("--embedding_dim", default= 16, type=int,help="number of embeddind dimension")
parser.add_argument("--max_length", default= 120, type=int,help="max word length in sentence")
parser.add_argument("--batch_size", default=128, type=int,help="batch size")
parser.add_argument("--num_epochs", default=5, type=int,help="number of training steps")
parser.add_argument("--early_stopping_criteria", default=2, type=int,help="early stop criteria")
parser.add_argument("--dropout", default=0.3, type=float,help="dropout percentage")
parser.add_argument("--model_storage", default="model_storage/lstm", type=str, help="model_storange")

def run_training(argv):
    with mlflow.start_run(mlflow.set_experiment("olist_sentiment_analisis")):
        args = parser.parse_args(argv[1:])

        clear_fn = clear_data_fn(column_text=COLUMN_TEXT, column_label=COLUMN_LABEL)
        df = load_from_file(args.path_dataset, clear_fn = clear_fn)
        LOG.info(f'Dataset loaded from {args.path_dataset} with {len(df)} observations.')

        tokenizer = fit_tokenizer(np.array(df['text'].values),args.vocab_size)
        LOG.info(f'Fit tokenization and creating vocabulary')

        transform_sequence = transform_to_sequence_fn(tokenizer=tokenizer, max_length=args.max_length)

        split_ds = split_ds_fn()

        train_dataset, val_dataset, test_dataset = preprocess_fn(df = df,
                                                                transfom_to_seq_fn = transform_sequence,
                                                                split_ds_fn = split_ds)
        LOG.info(f'Apply preprocess into dataset')

        model:tf.keras.Model = lstm.build(args)
        LOG.info(f'Build neural network lstm with {model.count_params()} parameters.')


        earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.early_stopping_criteria)
        tensorflowBoardCallback = ExtendedTensorBoard(test_dataset=test_dataset, logs_dir=MODEL_LOG_PATH)
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

        train_dataset = train_dataset.batch(32)
        val_dataset = val_dataset.batch(32)
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            validation_steps=10,
            epochs=args.num_epochs,
            callbacks= callbacks,
        )
        LOG.info(f'Fit model for {args.num_epochs}')

        # Evaluate the model
        print("Evaluate")
        result = model.evaluate(test_dataset.batch(32))
        print(dict(zip(model.metrics_names, result)))

