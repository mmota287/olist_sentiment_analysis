from argparse import Namespace


args = Namespace(
    dataset_csv = 'https://raw.githubusercontent.com/MarcosMota/AnaliseDeSentimento/master/dataset/olist_order_reviews_dataset.csv',
    train_split = 0.8,
    random_state = 42,
    vocab_size = 10000,
    embedding_dim = 16,
    max_length = 120,
    batch_size=128,
    num_epochs=5,
    early_stopping_criteria=2,
    dropout_p=0.3,
    model_storage="model_storage/lstm",
)

