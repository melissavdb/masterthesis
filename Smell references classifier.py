# Import packages
import pandas as pd
import re
import pickle
import numpy as np

from numpy.core.defchararray import find
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.tokenization import SciSpacyTokenizer
from ipywidgets import IntProgress
from sklearn.metrics import accuracy_scorefrom sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Import dataset
df_text = pd.read_excel("Data medical household - Frame elements new.xlsx", sheet_name="Text")
df = pd.read_excel("Data medical household - Frame elements new.xlsx", sheet_name="Source")

# Define new dataframe
df_sent = df_text

# Create unigrams 
unigrams  = (
    df_text['text'].str.lower()
                .str.replace(r'[^a-z\s]', '')
                .str.split(expand=True)
                .stack())

# Concatenate all series vertically
df_text = pd.concat([unigrams]).dropna().reset_index(drop=False)

# Rename columns
df_text.rename(columns = {'level_0':'row', 0:'word'}, inplace=True)

# Merge dataframes on row
df_text = pd.merge(df_text[['row', 'word']],
           df_sent[['row', 'text']],
           on = 'row',
           how = 'left')

# Transform text column
df_text['text'] = df_text['text'].str.lower()
df_text['text'] = df_text['text'].str.replace(r'[^a-z\s]', '')

# Find start position of word in text
a = df_text.text.values.astype(str)
b = df_text.word.values.astype(str)
df_text = df_text.assign(pos=find(a, b))

# Drop duplicatesfrom sklearn.metrics import precision_recall_fscore_support
df = df.drop_duplicates(subset = ['word', 'text', 'label'])

df["label"] = df["label"].astype(str)

# Replace strings
df['label'] = df['label'].str.replace('1','SOURCE')
df['label'] = df['label'].str.replace('2','EVOKED')
df['label'] = df['label'].str.replace('3','QUALITY')

# Transform text column
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r'[^a-z\s]', '')

# Find start position of word in text
a = df.text.values.astype(str)
b = df.word.values.astype(str)
df = df.assign(pos=find(a, b))

# Merge dataframes on text and position
df = pd.merge(df_text[['word', 'text', 'pos']],
           df[['text', 'pos', 'label']],
           on = ['text', 'pos'],
           how = 'left').reset_index()

# Fill NaN with zero
df['label'] = df['label'].fillna(0)

# Replace label with BIO scheme tags
df['annotation'] = np.where(df['label'] == 0, 'O', 'B-')
df.loc[(df['annotation'].shift(+1)=='B-') & (df['label'].shift(+1)==df['label']), 'annotation'] = 'I-'

df['ner'] = df['annotation'].astype(str) + df['label'].astype(str)
df['ner'] = df['ner'].str.replace(r'O0', 'O')

# Create blank rows after every sentence
indices = df.loc[df['text'].shift(+1) != df['text']].index.tolist()
rows_ = dict.fromkeys(df.columns.tolist(),'')

df = pd.DataFrame(np.insert(df.values, [x for x in indices],
                   values=list(rows_.values()), 
                   axis=0),columns=rows_.keys())

# Define final dataframe
df_final = df[['word', 'ner']].copy()

# Show dataframe
df_final.head()

# Save dataframe
df_final.to_csv(r'dev.txt', header=None, index=None, sep=' ', mode='a')

# Define columns
columns = {0 : 'text', 1 : 'ner'}

# Define directory
data_folder = '/Users/melissavandenbovenkamp/Documents/Master Data Science/Scriptie/Code/Flair/'

# Initialize corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              #train_file = 'train.text',
                              dev_file = 'dev.txt',
                              test_file = 'test.txt')

# Show length and first sentence of corpus
print(len(corpus.dev))
print(corpus.dev[0].to_tagged_string('ner'))

# Define tag to predict
tag_type = 'ner'

# Make tag dictionary from corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# Define embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
    FlairEmbeddings('nl-forward'),
    FlairEmbeddings('nl-backward'),
    TransformerWordEmbeddings('wietsedv/bert-base-dutch-cased', allow_long_sentences=True)
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# Initisalise sequence tagger
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True)

# Initisalise model
trainer : ModelTrainer = ModelTrainer(tagger, corpus)
    
trainer.train('/Users/melissavandenbovenkamp/Documents/Master Data Science/Scriptie/Code/Flair',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10,
              train_with_dev=True,
              checkpoint=True
              )

# Show evaluation of model
result, _ = tagger.evaluate(corpus.test)
print(result.detailed_results)

# Import dataset
df = pd.read_csv('data/test.tsv', sep='\t', header=None)

# Split dataset into multiple columns
df[[0, 1]] = df[0].str.split(' ', 1, expand=True)
df[[1, 2]] = df[1].str.split(' ', 1, expand=True)

# Replace subtstrings
df[1] = df[1].str.replace(r'B-', '')
df[2] = df[2].str.replace(r'B-', '')
df[1] = df[1].str.replace(r'I-', '')
df[2] = df[2].str.replace(r'I-', '')

# Create arrays out of dataframe
Y_test = df[[1]].to_numpy()
L_test = df[[2]].to_numpy()

# Calculate micro-averaged precision, recall, and F1-score
precision_recall_fscore_support(Y_test, L_test, average='micro', labels=['SOURCE', 'EVOKED', 'QUALITY'])

