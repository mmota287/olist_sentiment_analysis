from abc import ABC, abstractmethod
import tensorflow as tf
import time

class BaseModel(object):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, config):
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim

        self.name = f'{config.project}-{int(time.time())}'

        self.dropout = config.dropout
        self.dense = config.dense

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass