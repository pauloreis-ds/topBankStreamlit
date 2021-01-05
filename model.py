import pickle
import pandas as pd
from sklearn.metrics import recall_score


def _load_model():
    with open('./model/topBankModel.pkl', 'rb') as file:
        return pickle.load(file)


def _preprocess_data(raw_data):
    raw_data = pd.get_dummies(raw_data, prefix=['geography'], columns=['geography'])
    raw_data = pd.get_dummies(raw_data, prefix=['gender'], columns=['gender'] )

    data = raw_data.drop(columns=['exited', 'customer_id', 'surname', 'has_cr_card', 'tenure', 'total_revenue'])
    label = raw_data['exited']
    return data, label


class Model:

    def __init__(self):
        self.model = _load_model()
        self.label = None
        self.prediction = None
        self.prediction_probability = None

    def predict(self, raw_data):
        data, self.label = _preprocess_data(raw_data)
        self.prediction = self.model.predict(data)
        return self.prediction

    def predict_probability(self, raw_data):
        data, self.label = _preprocess_data(raw_data)
        self.prediction_probability = self.model.predict_proba(data)[:, 1]
        return self.prediction_probability

    def get_recall(self):
        return recall_score(self.label, self.prediction)