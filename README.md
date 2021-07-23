# Olist Analysis Sentiment

This project uses Tensorflow to implement a Neural LSTM Network that can classify sentiments present in purchase reviews from the top marketplaces in Brazil.

You can get about the project by reading the article on my [blog](https://marcosmota.com/tutorial-an%C3%A1lise-de-sentimentos-usando-redes-neurais-recorrentes-lstm-e-tensorflow-5e3f26c2e8db).

## Built With
This project was built with the following frameworks and libraries
- Tensorflow
- MLFlow
Attempt |
- Numpy

## Usage

### Installation
```
conda env create -f conda.yaml
conda activate olist_sentiment_analisis
```

### Training Model
```
python main.py <options>
```
#### Options

Options | Description | Default
--- | --- | --- 
--path_dataset | dataset path | 
--lr | learning rate | 0.001
--train_split | number of divide dataset train | 0.8
--random_state | random state |  42
--vocab_size | vocabulary size |  10000
--embedding_dim | number of embeddind dimension |  16
--max_length | max word length in sentence |  120
--batch_size | batch size | 128
--num_epochs | number of training steps | 5
--early_stopping_criteria | early stop criteria | 2
--dropout | dropout percentage | 0.3
--model_storage | model_storange | model_storage/lstm

### MLFlow UI
```
mlflow ui
```

### Server model with MLFlow

```
mlflow models serve -m runs:/1addb128068e4cff8292f671dfab48fe/model
```


