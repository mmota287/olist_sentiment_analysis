import io
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os
import numpy as np

def load_tokenizer(vectorizer_file: str = 'resources/vectorizer.json',  vocab_size: int = None) -> Tokenizer:
    """Load tokenizer from file or Create a new with [vocab_size]

    Args:
        vectorizer_file (str, optional): Vectorizer file path. Defaults to 'resources/vectorizer.json'.
        vocab_size (int, optional): Vocabulary Size. Defaults to None.

    Raises:
        Exception: If notfound [vectorizer_file] and not informed [vocab_size]

    Returns:
        Tokenizer: Tensorflow Tokenizer
    """
    if os.path.exists(vectorizer_file) :
        with open(vectorizer_file) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
            return tokenizer
    else:
        if vocab_size is None:
            raise Exception('vocab_size', 'Vocabulary size is required to build a new tokenizer')
        return Tokenizer(num_words = vocab_size, oov_token="<OOV>")

def save_tokenizer(tokenizer: Tokenizer, vectorizer_file: str = 'resources/vectorizer.json') -> None:
    """Save tokenizer in [vectorizer_file]

    Args:
        tokenizer (Tokenizer): Tensorflow Tokenizer
        vectorizer_file (str, optional): Vectorizer file path. Defaults to 'resources/vectorizer.json'.
    """
    
    tokenizer_json = tokenizer.to_json()
    with io.open(vectorizer_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def fit_tokenizer(sentences: np.array, vocab_size: int) -> Tokenizer:
    tokenizer = load_tokenizer(vocab_size=vocab_size)
    tokenizer.fit_on_texts(sentences)

    save_tokenizer(tokenizer)

    return tokenizer
