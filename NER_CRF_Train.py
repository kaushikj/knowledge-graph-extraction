# This code is for training the NER algorithm using the Conditional Random Forest(CRF) model.

# Import the packages
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

import sklearn_crfsuite

# Read the NER Dataset
df = pd.read_csv('data.csv', encoding='latin1')
df = df.fillna(method='ffill')

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]


sep_sent_func = lambda data: [(word, pos, tag) for word, pos, tag in zip(data['Word'].values.tolist(), 
                                                                    data['POS'].values.tolist(), 
                                                                    data['Tag'].values.tolist())]

final_df = df.groupby('Sentence #').apply(sep_sent_func)
sentences = [sentence for sentence in final_df]

# Training the CRF model
X_data = np.array([sent2features(sentence) for sentence in sentences])
y_label = np.array([sent2labels(sentence) for sentence in sentences])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.10, random_state=42)

# Define the CRF model
crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                           c1=0.1,
                           c2=0.1,
                           max_iterations=100,
                           all_possible_transitions=True,
                           verbose=True)
# Fit the CRF model on train data
crf.fit(X_train, y_train)

# Saving the trained CRF model
joblib.dump(crf, 'Models/ner_model_trained.pkl')

from sklearn_crfsuite import metrics as crf_metrics
y_pred = crf.predict(X_test)
y_pred_train = crf.predict(X_train)
#print (crf_metrics.sequence_accuracy_score(y_test, y_pred))

