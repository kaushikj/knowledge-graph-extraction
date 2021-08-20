# Knowledge Graph Extraction

## Description
A vast majority of data exists as unstructured text. Therefore, it is extremely important to harness its value by extracting meaningful information from it. Knowledge graph extraction is a technique that identifies the entities and relations between them. This helps us convert unstructured text into structured information, and this information can be leveraged by AI to build models for Question-Answering.

![Flowchart image](https://github.com/kaushikj/knowledge-graph-extraction/blob/main/Screenshots/flowchart.png)

We break down the project into two parts. We will build each model separately with a separate dataset.

1. Named Entity Recognition: First part of the project is to identify the different named entities from text data. This is built using 2 models - Bi-LSTM and CRF. 
2. Entity relation extraction: From the entities that were identified we would find the relationship between the entities and output a knowledge graph with the relations. This is built using BERT ecoder and Neural Network.


![output image](https://github.com/kaushikj/knowledge-graph-extraction/blob/main/Screenshots/Bi-LSTM_Result.png)

## Setup

1. Run sh setup.sh
2. pip install -r requirements.text


## Running the code
streamlit run app.py


## Developers
- Kaushik Jeyaraman - kaushikjjj@gmail.com
- Neeraja Neelakantan
- Hridya Divakaran
