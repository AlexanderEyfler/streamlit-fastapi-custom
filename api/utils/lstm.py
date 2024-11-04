import joblib
import streamlit as st
import numpy as np
# import pandas as pd
from typing import Tuple
import torch
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from time import time
import re
import unicodedata


# Определяем путь к директории скрипта
script_path = Path(__file__).resolve()
script_dir = script_path.parent

## Пути для LSTM
##----------------------------------------------------------------------------
path_lstm = script_dir.parent / 'data' / 'lstm_model.pth'
path_int_to_vocab = script_dir.parent / 'data' / 'int_to_vocab.pkl'
path_vocab_to_int = script_dir.parent / 'data' / 'vocab_to_int.pkl'
# path_f1_lstm = script_dir.parent / 'data' / 'lstm' / 'val_f1.pkl'
path_stopwords = script_dir.parent / 'data' / 'stop_words.pkl'
##----------------------------------------------------------------------------

# Загрузка словарей
stop_words = joblib.load(path_stopwords)
vocab_to_int = joblib.load(path_vocab_to_int)
int_to_vocab = joblib.load(path_int_to_vocab)


VOCAB_SIZE = len(vocab_to_int) + 1
EMBEDDING_DIM = 64
SEQ_LEN = 64
BATCH_SIZE = 64
HIDDEN_SIZE = 32

## Загрузка LSTM
##----------------------------------------------------------------------------
# @st.cache_resource
def load_lstm():
    # # Загрузка словарей
    # vocab_to_int = joblib.load(path_vocab_to_int)
    # int_to_vocab = joblib.load(path_int_to_vocab)

    # Загрузка модели
    model_lstm = LSTMConcatAttention()
    model_lstm.load_state_dict(torch.load(path_lstm, map_location=torch.device('cpu')))
    model_lstm.eval()

    # Загрузка метрики
    # val_f1 = joblib.load(path_f1_lstm)

    return model_lstm # , vocab_to_int, int_to_vocab # , val_f1

def load_vocabs():
    # Загрузка словарей
    vocab_to_int = joblib.load(path_vocab_to_int)
    int_to_vocab = joblib.load(path_int_to_vocab)
    return vocab_to_int, int_to_vocab

## Определение класса Attention
##----------------------------------------------------------------------------
class ConcatAttention(nn.Module):
    def __init__(
            self, 
            hidden_size: int = HIDDEN_SIZE
            ) -> None:

        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.align  = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tanh   = nn.Tanh()

    def forward(
            self, 
            lstm_outputs: torch.Tensor,
            final_hidden: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:

        att_weights = self.linear(lstm_outputs)
        att_weights = torch.bmm(att_weights, final_hidden.unsqueeze(2))
        att_weights = F.softmax(att_weights, dim=1)
        cntxt = torch.bmm(lstm_outputs.transpose(1, 2), att_weights)
        concatted   = torch.cat((cntxt, final_hidden.unsqueeze(2)), dim=1)
        att_hidden  = self.tanh(self.align(concatted.squeeze(-1)))
        return att_hidden, att_weights
##----------------------------------------------------------------------------


## Определение класса LSTM
##----------------------------------------------------------------------------
class LSTMConcatAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.attn = ConcatAttention(HIDDEN_SIZE)
        self.clf = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        outputs, (h_n, _) = self.lstm(embeddings)
        att_hidden, att_weights = self.attn(outputs, h_n.squeeze(0))
        out = self.clf(att_hidden)
        return out, att_weights
##----------------------------------------------------------------------------

## Определенеие функции паддинга (из лекции)
##----------------------------------------------------------------------------
def padding(review_int: list, seq_len: int) -> np.array: # type: ignore
    """Make left-sided padding for input list of tokens

    Args:
        review_int (list): input list of tokens
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

    Returns:
        np.array: padded sequences
    """
    features = np.zeros((len(review_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)

    return features
##----------------------------------------------------------------------------

## Функция предобработки текста
##----------------------------------------------------------------------------
def custom_clean(text):
    text = text.lower()  # нижний регистр
    text = re.sub(r'http\S+', " ", text)  # удаляем ссылки
    text = re.sub(r'@\w+',' ',text)  # удаляем упоминания пользователей
    text = re.sub(r'#\w+', ' ', text)  # удаляем хэштеги
    text = re.sub(r'\d+', ' ', text)  # удаляем числа

    # Удаляем эмодзи и специальные символы © и ®
    def remove_emojis(text):
        return ''.join(
            c for c in text
            if not unicodedata.category(c) in ('So', 'Cn', 'Cs')
        )
    text = remove_emojis(text)

    # Удаляем пунктуацию и специальные символы
    def remove_punctuation(text):
        return ''.join(c for c in text if not unicodedata.category(c).startswith('P'))
    text = remove_punctuation(text)

    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Удаляем все слова на английском языке
    text = re.sub(r'\b[a-z]+\b', ' ', text)

    # Удаляем стоп-слова на русском языке
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text
##----------------------------------------------------------------------------

## Определение функци препроцессинга одной строки (из лекции)
##----------------------------------------------------------------------------
def preprocess_single_string(
    input_string: str, 
    seq_len: int, 
    vocab_to_int: dict,
    verbose : bool = False
    ) -> Tensor:
    """Function for all preprocessing steps on a single string

    Args:
        input_string (str): input single string for preprocessing
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros
        vocab_to_int (dict, optional): word corpus {'word' : int index}. Defaults to vocab_to_int.

    Returns:
        list: preprocessed string
    """

    preprocessed_string = custom_clean(input_string)
    result_list = []
    for word in preprocessed_string.split():
        try: 
            result_list.append(vocab_to_int[word])
        except KeyError as e:
            if verbose:
                print(f'{e}: not in dictionary!')
            pass
    result_padded = padding([result_list], seq_len)[0]

    return Tensor(result_padded)
##----------------------------------------------------------------------------


## Предсказание LSTM
##----------------------------------------------------------------------------
def predict_lstm(review: str, seq_len: int, model, vocab_to_int):
    start_time = time()

    # Предобработка
    inp = preprocess_single_string(review, seq_len, vocab_to_int)
    inp = inp.long().unsqueeze(0)  # Добавляем размерность батча

    # Предсказание
    with torch.inference_mode():
        pred, att_scores = model(inp)

    end_time = time()
    prediction_time = end_time - start_time

    # Получаем предсказанный класс
    predicted_class = torch.argmax(pred, dim=1).item()
    # Получаем вероятности для каждого класса
    probabilities = torch.softmax(pred, dim=1).squeeze().cpu().numpy()
    
    print(f"Model output (logits): {pred}")
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")

    return predicted_class, probabilities, prediction_time
