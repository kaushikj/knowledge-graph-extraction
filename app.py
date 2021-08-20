# This code runs the application in Streamlit.
# Run the command "streamlit run app.py" in terminal to run. 

# Import the necessary packages
import streamlit as st
import numpy as np
import pandas as pd
import graphviz
from annotated_text import annotated_text
import tensorflow
from tensorflow import keras
import string
import os
import nltk
import joblib
import random
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('stopwords')
from bs4 import BeautifulSoup
import collections
from collections import Counter
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import json
import import_ipynb
from relation_extractor import get_model_bert_wiki80
from itertools import combinations
import graphviz
from PIL import Image

MIN_THRESHOLD_RE_PREDICTION = 0.90

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger')
from keras.callbacks import ModelCheckpoint

from random import randrange

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)
from PIL import Image

# Read the NER Dataset
data = pd.read_csv('data.csv', encoding='latin1')

st.title("Knowledge Graph Extraction from Text")

def example(color1, color2, color3, content):
     st.markdown(f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:16px;border-radius:2%;">{content}</p>', unsafe_allow_html=True)
text222 = "For identifying entities and relations from Text."
color1 = "lightgrey"
color2 = "lightpink"
color3 = "black"
example(color1, color2, color3, text222)

def example1(color3, content):
    st.markdown(f'<p style=color:{color3};font-size:16px;font-weight:bold;border-radius:2%;">{content}</p>', unsafe_allow_html=True)
text222 = "Hello World"
color1 = "#fea"
color2 = "#faa"
color3 = "#8ef"
color3 = "black"
st.write("")
st.write("")
option = st.selectbox(
 'Select the ML model to identifying Entities',
('CRF (SkLearn)', 'Bi-LSTM (Keras)'))

st.write("")
st.write("")
_input = st.text_area("Enter Text and press 'Submit' to extract entities and relations : ")

NER_positions_list = []

## Model 1 - Bi-LSTM (Keras)
if option == 'Bi-LSTM (Keras)':

    if(st.button('Submit')):

        st.write("Scroll down to see the annotated text and generated Knowledge Graph.")

        def createIndexDictionary(dataset, type):
            tok2idxArr = {}
            idx2tokArr = {}
            
            if type == 'token':
                vocab = list(set(dataset['Word'].to_list()))
            else:
                vocab = list(set(dataset['Tag'].to_list()))
            
            idx2tokArr = {idx:tok for  idx, tok in enumerate(vocab)}
            tok2idxArr = {tok:idx for  idx, tok in enumerate(vocab)}
            return tok2idxArr, idx2tokArr

        token2idx, idx2token = createIndexDictionary(data, 'token')
        tag2idx, idx2tag = createIndexDictionary(data, 'tag')

        data['Word_idx'] = data['Word'].map(token2idx)
        data['Tag_idx'] = data['Tag'].map(tag2idx)
        words = list(set(data["Word"].values))
        tags = list(set(data["Tag"].values))

        data_fillna = data.fillna(method='ffill', axis=0)
        grouped_data = data_fillna.groupby(['Sentence #'],as_index=False
                                    )['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))

        def split_padded_dataset(grouped_data, data):
            n_token = len(list(set(data['Word'].to_list())))
            n_tag = len(list(set(data['Tag'].to_list())))
            
            tokens = grouped_data['Word_idx'].tolist()
            maxlen = max([len(s) for s in tokens])
            pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value= n_token - 1)
            
            tags = grouped_data['Tag_idx'].tolist()
            pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int32', padding='post', value= tag2idx["O"])
            
            n_tags = len(tag2idx)
            pad_tags = [to_categorical(i, num_classes=n_tags) for i in pad_tags]
            
            
            train_tokens, test_tokens, train_tags, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, train_size=0.9, random_state=2020)
            return train_tokens, test_tokens, train_tags, test_tags

        train_tokens, test_tokens, train_tags, test_tags = split_padded_dataset(grouped_data, data)
        input_dim = len(list(set(data['Word'].to_list())))+1
        output_dim = 64
        input_length = max([len(s) for s in grouped_data['Word_idx'].tolist()])
        n_tags = len(tag2idx)

        from tensorflow.keras import Model,Input
        from tensorflow.keras.layers import LSTM,Embedding,Dense
        from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional

        def createLSTMModel():
            input_word = Input(shape=(input_length,))
            model = Embedding(input_dim=input_dim,output_dim=output_dim,input_length=input_length)(input_word)
            model = SpatialDropout1D(0.1)(model)
            model = Bidirectional(LSTM(units=100,return_sequences=True,recurrent_dropout=0.1))(model)
            out = TimeDistributed(Dense(n_tags,activation='softmax'))(model)
            model = Model(input_word,out)
            model.summary()
            
            return model

        import tensorflow as tf

        def train_model(X, y, model):   
            hist = model.fit(X, y, batch_size=32, verbose=1, epochs=3, validation_split=0.2)
            
        nerModel = createLSTMModel()
        nerModel.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])
        train_model(train_tokens, np.array(train_tags), nerModel)
        
        def cleanText(test_sentence): 
            cleantext = BeautifulSoup(test_sentence).get_text()
            cleantext = re.sub("\d+", " ", cleantext) #remove digits
            cleantext = re.sub("\s+", " ", cleantext) #remove extra spaces

            cleantext = word_tokenize(cleantext)
            return cleantext

        test_sentence = _input
        cleaned_test_sentence = cleanText(test_sentence)
        x_test_sent = pad_sequences(sequences=[[token2idx.get(w, 0) for w in cleaned_test_sentence]],padding="post", value=0, maxlen=input_length)

        p = nerModel.predict(np.array([x_test_sent[0]]))
        p = np.argmax(p, axis=-1)
        NER_tuple = []
     
        for w, pred in zip(cleaned_test_sentence, p[0]):
            NER_tuple.append((w, tags[pred]))

        entities = []
        join_entity = ''
        temp_entity = None
        all_entities = []
        
        for i in NER_tuple:
            term, tag = i[0], i[1]
            if tag != 'O':
                join_entity = ' '.join([join_entity, term]).strip()
                temp_entity = (join_entity, tag)
            else:
                if temp_entity:
                    entities.append(temp_entity)
                    all_entities.append(temp_entity)
                    join_entity = ''
                    temp_entity = None
                else:
                    all_entities.append((term, tag))

        NER_positions_list  = [[(i, i+1) for i,x in enumerate(all_entities) if x[1] != 'O']]
        
        # Displaying the annotated text with the identified named entities
        annotated_text_list = []
        color_list = ["#fea", "#faa", "#8ef", "lightgrey"]

        example1(color3, "The annotated text tagged with the identified ENTITIES is given below : ")
        for entity, tag in all_entities:
            if tag == 'O':
                annotated_text_list.append(" " + entity)
            else:
                annotated_text_list.append((" " + entity, tag, random.choice(color_list)))
        
        annotated_text(*annotated_text_list) 

## Model 2 - Conditional Random Forest(SKlearn)
elif option == 'CRF (SkLearn)': 

    if(st.button('Submit')):

        st.write("Scroll down to see the annotated text and generated Knowledge Graph. ")

        # Model 2: Load the trained CRF model
        crf = joblib.load('Models/ner_model_trained.pkl')
        text = _input
        text = re.sub(r'\n', '', text)

        words_tokens = nltk.word_tokenize(text)
        words_posit = nltk.pos_tag(words_tokens)

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

        words_features = [sent2features(words_posit)]
        pred_labels = crf.predict(words_features)
        document_labels = pred_labels[0]
        input_text = [(token, tag) for token, tag in zip(words_tokens, document_labels)]

         #adding all entities
        entities = []
        join_entity = ''
        temp_entity = None
        all_entities = []

        for term, tag in input_text:
            if tag != 'O':
                join_entity = ' '.join([join_entity, term]).strip()
                temp_entity = (join_entity, tag)
            else:
                if temp_entity:
                    entities.append(temp_entity)
                    all_entities.append(temp_entity)
                    join_entity = ''
                    temp_entity = None
                else:
                    all_entities.append((term, tag))
        input_text = all_entities
        annotated_text_list = []
        color_list = ["#fea", "#faa", "#8ef", "lightgrey"]
        example1(color3, "The annotated text tagged with the identified ENTITIES is given below : ")

        for term, tag in all_entities:
            if tag == 'O':
                annotated_text_list.append(" " + term)
            else:
                annotated_text_list.append((" " + term, tag, random.choice(color_list)))

        annotated_text(*annotated_text_list) 

        NER_positions_list  = [[(i, i+1) for i,x in enumerate(all_entities) if x[1] != 'O']]
        # NER_positions_list = [[(0, 2), (8, 9), (14, 16)]]

# Relation Extraction from entities
if len(NER_positions_list) != 0:
    
    re_model = get_model_bert_wiki80()
    text =  _input
    tokens = [entity for entity, tag in all_entities]

    node_relations = []
    def predict_Relation(tokens, ner_pair_0, ner_pair_1):

        r1_rel, r1_score = re_model.infer({'token': tokens, 'h': {'pos': ner_pair_0}, 't': {'pos': ner_pair_1}})
        head = ' '.join([tokens[i] for i in range(ner_pair_0[0], ner_pair_0[1])])
        tail = ' '.join([tokens[i] for i in range(ner_pair_1[0], ner_pair_1[1])])
        node_relation_r = None
        if r1_score > MIN_THRESHOLD_RE_PREDICTION:
            node_relation_r = {
                'head': head,
                'tail': tail,
                'relation': r1_rel
            }

        r2_rel, r2_score = re_model.infer({'token': tokens, 'h': {'pos': ner_pair_1}, 't': {'pos': ner_pair_0}})
        head = ' '.join([tokens[i] for i in range(ner_pair_1[0], ner_pair_1[1])])
        tail = ' '.join([tokens[i] for i in range(ner_pair_0[0], ner_pair_0[1])])
        node_relation_l = None
        if r2_score > MIN_THRESHOLD_RE_PREDICTION:
            node_relation_l = {
                'head': head,
                'tail': tail,
                'relation': r2_rel
            }

        if node_relation_r and node_relation_l:
            if r1_score > r2_score:
                node_relations.append(node_relation_r)
            else:
                node_relations.append(node_relation_l)
        elif node_relation_r:
                node_relations.append(node_relation_r)
        elif node_relation_l:
                node_relations.append(node_relation_l)

    # Different combinations of pairs of entities
    combin_NER_List = list(combinations(NER_positions_list[0],2))
    for i in combin_NER_List:
        predict_Relation(tokens, i[0], i[1])

    nodes = set()
    edges = {}

    for i in node_relations:
        if i.get('head') != i.get('tail'):
            nodes.add(i.get('head'))
            nodes.add(i.get('tail'))
        
            edges[(i.get('head'), i.get('tail'))] = i.get('relation')

    # Display the Knowledge Graph using Graphviz
    def KnowledgeGraph(o):
        g = graphviz.Digraph(format='png') 
        for (node1,node2), weight in o.items():
            g.node(node1, style='filled', fillcolor='cyan')
            g.node(node2, style='filled', fillcolor='cyan')
            g.edge(str(node1), str(node2), label=str(weight), color = 'black')
        g.attr(size='12')
        g.render('knowledge_graph')
        return g

    KnowledgeGraph(edges)
    final_graph = Image.open(r"knowledge_graph.png")
    st.header("Knowledge Graph depicting the relationship between entities")
    st.image(final_graph, use_column_width=False )


# Reference : https://eli5.readthedocs.io/en/latest/tutorials/sklearn_crfsuite.html