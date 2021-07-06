# Import packages
import pandas as pd
import re
import numpy as np
import spacy

from snorkel.labeling import LabelingFunction
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelModel
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import filter_unlabeled_dataframe

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy.core.defchararray import find

# Import dataset
df = pd.read_excel()
df2 = pd.read_excel()

# Import datasets
df_train = df
df_test = df2

# Define test labels
Y_test = df_test.label.values

# Show dataframe
df_test.head()

# Create unigrams 
unigrams  = (
    df_train['text'].str.lower()
                .str.replace(r'[^a-z\s]', '')
                .str.split(expand=True)
                .stack())

# Create bigrams, trigrams, fourgrams, and fivegrams by concatenating unigram columns
bigrams = unigrams + ' ' + unigrams.shift(-1)
trigrams = bigrams + ' ' + unigrams.shift(-2)
fourgrams = trigrams + ' ' + unigrams.shift(-3)
fivegrams = fourgrams + ' ' + unigrams.shift(-4)

# Concatenate all series vertically
df_train = pd.concat([unigrams, bigrams, trigrams, fourgrams, fivegrams]).dropna().reset_index(drop=False)

# Rename columns
df_train.rename(columns = {'level_0':'row', 'level_1':'row_word', 0:'word'}, inplace=True)

# Show dataframe
df_train

# Merge dataframes on row numbers
df_train = pd.merge(df_train[['row', 'row_word', 'word']],
           df[['row', 'text']],
           on = 'row',
           how = 'left')

# Show dataframe
df_train.head()

# Define labels
NO_SMELL = 0
SOURCE = 1
EVOKED = 2
QUALITY = 3

# Define regular expressions
@labeling_function()
def lf_regex_source_keyword(x):
    return SOURCE if re.search(r"^\bgeur\b$|^\breuck\b$|^\breuk\b$|^\bstanck\b$|^\bstank\b$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_van(x):
    return SOURCE if re.search(r"^\bgeur van [a-z]{1,20}$|^\breuck van [a-z]{1,20}$|^\breuk van [a-z]{1,20}$|^\bstanck van [a-z]{1,20}$|^\bstank van [a-z]{1,20}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_van_two(x):
    return SOURCE if re.search(r"^\bgeur van [a-z]{1,20} [a-z]{1,20}$|^\breuck van [a-z]{1,20} [a-z]{1,20}$|^\breuk van [a-z]{1,20} [a-z]{1,20}$|^\bstanck van [a-z]{1,20} [a-z]{1,20}$|^\bstank van [a-z]{1,20} [a-z]{1,20}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_van_three(x):
    return SOURCE if re.search(r"^\bgeur van [a-z]{1,20} [a-z]{1,20} [a-z]{1,20}$|^\breuck van [a-z]{1,20} [a-z]{1,20} [a-z]{1,20}$|^\breuk van [a-z]{1,20} [a-z]{1,20} [a-z]{1,20}$|^\bstanck van [a-z]{1,20} [a-z]{1,20} [a-z]{1,20}$|^\bstank van [a-z]{1,20} [a-z]{1,20} [a-z]{1,20}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_heeft(x):
    return SOURCE if re.search(r"^[a-z]{1,20} heeft (.*?)geur$|^[a-z]{1,20} heeft (.*?)reuck$|^[a-z]{1,20} heeft (.*?)reuk$|^[a-z]{1,20} heeft (.*?)stanck$|^[a-z]{1,20} heeft (.*?)stank$|^[a-z]{1,20} hebben (.*?)geur$|^[a-z]{1,20} hebben (.*?)reuck$|^[a-z]{1,20} hebben (.*?)reuk$|^[a-z]{1,20} hebben (.*?)stanck$|^[a-z]{1,20} hebben (.*?)stank$|^[a-z]{1,20} had[a-z]{0,3} (.*?)geur$|^[a-z]{1,20} had[a-z]{0,3} (.*?)reuck$|^[a-z]{1,20} had[a-z]{0,3} (.*?)reuk$|^[a-z]{1,20} had[a-z]{0,3} (.*?)stanck$|^[a-z]{1,20} had[a-z]{0,3} (.*?)stank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_geeft(x):
    return SOURCE if re.search(r"^[a-z]{1,20} geeft (.*?)geur$|^[a-z]{1,20} geeft (.*?)reuck$|^[a-z]{1,20} geeft (.*?)reuk$|^[a-z]{1,20} geeft (.*?)stanck$|^[a-z]{1,20} geeft (.*?)stank$|^[a-z]{1,20} geven (.*?)geur$|^[a-z]{1,20} geven (.*?)reuck$|^[a-z]{1,20} geven (.*?)reuk$|^[a-z]{1,20} geven (.*?)stanck$|^[a-z]{1,20} geven (.*?)stank$|^[a-z]{1,20} gaf (.*?)geur$|^[a-z]{1,20} gaf (.*?)reuck$|^[a-z]{1,20} gaf (.*?)reuk$|^[a-z]{1,20} gaf (.*?)stanck$|^[a-z]{1,20} gaf (.*?)stank$|^[a-z]{1,20} gaven (.*?)geur$|^[a-z]{1,20} gaven (.*?)reuck$|^[a-z]{1,20} gaven (.*?)reuk$|^[a-z]{1,20} gaven (.*?)stanck$|^[a-z]{1,20} gaven (.*?)stank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_heeft_geeft(x):
    return SOURCE if re.search(r"^de[a-z]{0,4} [a-z]{1,20} heeft$|^de[a-z]{0,4} [a-z]{1,20} hebben$|^de[a-z]{0,4} [a-z]{1,20} had[a-z]{0,3}$|^de[a-z]{0,4} [a-z]{1,20} geeft$|^de[a-z]{0,4} [a-z]{1,20} geven$|^de[a-z]{0,4} [a-z]{1,20} gaf$|^de[a-z]{0,4} [a-z]{1,20} gaven$|^het [a-z]{1,20} heeft$|^het [a-z]{1,20} hebben$|^het [a-z]{1,20} had[a-z]{0,3}$|^het [a-z]{1,20} geeft$|^het [a-z]{1,20} gaf$|^het [a-z]{1,20} gaven$|^een[a-z]{0,2} [a-z]{1,20} heeft$|^een[a-z]{0,2} [a-z]{1,20} had[a-z]{0,3}$|^een[a-z]{0,2} [a-z]{1,20} geeft$|^een[a-z]{0,2} [a-z]{1,20} gaf$", x.word, flags=re.I) and re.search(r" geur| reuck| reuk| stanck| stanck", x.text, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_heeft_geeft_two(x):
    return SOURCE if re.search(r"^de[a-z]{0,4} [a-z]{1,20} [a-z]{1,20} heeft$|^de[a-z]{0,4} [a-z]{1,20} [a-z]{1,20} hebben$|^de[a-z]{0,4} [a-z]{1,20} [a-z]{1,20} had[a-z]{0,3}$|^de[a-z]{0,4} [a-z]{1,20} [a-z]{1,20} geeft$|^de[a-z]{0,4} [a-z]{1,20} [a-z]{1,20} geven$|^de[a-z]{0,4} [a-z]{1,20} [a-z]{1,20} gaf$|^de[a-z]{0,4} [a-z]{1,20} [a-z]{1,20} gaven$|^het [a-z]{1,20} [a-z]{1,20} heeft$|^het [a-z]{1,20} [a-z]{1,20} hebben$|^het [a-z]{1,20} [a-z]{1,20} had[a-z]{0,3}$|^het [a-z]{1,20} [a-z]{1,20} geeft$|^het [a-z]{1,20} [a-z]{1,20} gaf$|^het [a-z]{1,20} [a-z]{1,20} gaven$|^een[a-z]{0,2} [a-z]{1,20} [a-z]{1,20} heeft$|^een[a-z]{0,2} [a-z]{1,20} [a-z]{1,20} had[a-z]{0,3}$|^een[a-z]{0,2} [a-z]{1,20} [a-z]{1,20} geeft$|^een[a-z]{0,2} [a-z]{1,20} [a-z]{1,20} gaf$", x.word, flags=re.I) and re.search(r" geur| reuck| reuk| stanck| stanck", x.text, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_is(x):
    return SOURCE if re.search(r"^[a-z]{1,20} is (.*?) geur$|^[a-z]{1,20} is (.*?) reuck$|^[a-z]{1,20} is (.*?) reuk$|^[a-z]{1,20} is (.*?) stanck$|^[a-z]{1,20} is (.*?) stank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_zijn(x):
    return SOURCE if re.search(r"^[a-z]{1,20} zijn (.*?) geur$|^[a-z]{1,20} zijn (.*?) reuck$|^[a-z]{1,20} zijn (.*?) reuk$|^[a-z]{1,20} zijn (.*?) stanck$|^[a-z]{1,20} zijn (.*?) stank$|^[a-z]{1,20} zyn (.*?) geur$|^[a-z]{1,20} zyn (.*?) reuck$|^[a-z]{1,20} zyn (.*?) reuk$|^[a-z]{1,20} zyn (.*?) stanck$|^[a-z]{1,20} zyn (.*?) stank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_is_zijn(x):
    return SOURCE if re.search(r"^de[a-z]{0,4} [a-z]{1,20} is$|^de[a-z]{0,4} [a-z]{1,20} zijn$|^de[a-z]{0,4} [a-z]{1,20} zyn$|^de[a-z]{0,4} [a-z]{1,20} was$|^de[a-z]{0,4} [a-z]{1,20} waren$|^het [a-z]{1,20} is$|^het [a-z]{1,20} zijn$|^het [a-z]{1,20} zyn$|^het [a-z]{1,20} was$|^het [a-z]{1,20} waren$|^een[a-z]{0,2} [a-z]{1,20} is$|^een[a-z]{0,2} [a-z]{1,20} zijn$|^een[a-z]{0,2} [a-z]{1,20} zyn$|^een[a-z]{0,2} [a-z]{1,20} was$|^een[a-z]{0,2} [a-z]{1,20} waren$", x.word, flags=re.I) and re.search(r" geur| reuck| reuk| stanck| stanck", x.text, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_dezer(x):
    return SOURCE if re.search(r"^geur de[a-z]{0,1}zea-z]{0,1} [a-z]{1,20}$|^reuck de[a-z]{0,1}ze[a-z]{0,1} [a-z]{1,20}$|^reuk de[a-z]{0,1}ze[a-z]{0,1} [a-z]{1,20}$|^stanck de[a-z]{0,1}ze[a-z]{0,1} [a-z]{1,20}$|^stank de[a-z]{0,1}ze[a-z]{0,1} [a-z]{1,20}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_der(x):
    return SOURCE if re.search(r"^geur de[a-z]{1} [a-z]{1,20}$|^reuck de[a-z]{1} [a-z]{1,20}$|^reuk de[a-z]{1} [a-z]{1,20}$|^stanck de[a-z]{1} [a-z]{1,20}$|^stank de[a-z]{1} [a-z]{1,20}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_der_two(x):
    return SOURCE if re.search(r"^geur de[a-z]{1} [a-z]{1,20} [a-z]{1,20}$|^reuck de[a-z]{1} [a-z]{1,20} [a-z]{1,20}$|^reuk de[a-z]{1} [a-z]{1,20} [a-z]{1,20}$|^stanck de[a-z]{1} [a-z]{1,20} [a-z]{1,20}$|^stank de[a-z]{1} [a-z]{1,20} [a-z]{1,20}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_welke(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} welke[a-z]{0,1} (.*?)geur$|^\b[a-z]{1,20} welke[a-z]{0,1} (.*?)reuck$|^\b[a-z]{1,20} welke[a-z]{0,1} (.*?)reuk$|^\b[a-z]{1,20} welke[a-z]{0,1} (.*?)stanck|^\b[a-z]{1,20} welke[a-z]{0,1} (.*?)stank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_welke_two(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} [a-z]{1,20} welke[a-z]{0,1} (.*?)geur$|^\b[a-z]{1,20} [a-z]{1,20} welke[a-z]{0,1} (.*?)reuck$|^\b[a-z]{1,20} [a-z]{1,20} welke[a-z]{0,1} (.*?)reuk$|^\b[a-z]{1,20} [a-z]{1,20} welke[a-z]{0,1} (.*?)stanck|^\b[a-z]{1,20} [a-z]{1,20} welke[a-z]{0,1} (.*?)stank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_aan(x):
    return SOURCE if re.search(r"^\bgeur[a-z]{0,5} aan (.*?)$|^\breuck[a-z]{0,5} aan (.*?)$|^\breuk[a-z]{0,5} aan (.*?)$|^\brieck[a-z]{0,5} aan (.*?)$|^\briek[a-z]{0,5} aan (.*?)$|$^\bstanck[a-z]{0,5} aan (.*?)$|$^\bstank[a-z]{0,5} aan (.*?)$|$^\bstinck[a-z]{0,5} aan (.*?)$|$^\bstink[a-z]{0,5} aan (.*?)$|^\bruik[a-z]{0,5} aan (.*?)$|^\bruyck[a-z]{0,5} aan (.*?)$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_adjective(x):
    return SOURCE if re.search(r"^\baromatiek[a-z]{0,4} [a-z]{1,20}$|^\baromatisch[a-z]{0,4} [a-z]{1,20}$|^\baromatijk[a-z]{0,4} [a-z]{1,20}$|^\bbalsamisch[a-z]{0,4} [a-z]{1,20}$|^\bbalsemend[a-z]{0,4} [a-z]{1,20}$|^\bbalsemiek[a-z]{0,4} [a-z]{1,20}$|^\bbalsemig[a-z]{0,4} [a-z]{1,20}$|^\bbalzemig[a-z]{0,4} [a-z]{1,20}$|^\bgeparfumeerd[a-z]{0,4} [a-z]{1,20}$|^\b[a-z]{0,2}geurig[a-z]{0,4} [a-z]{1,20}$|^\bgeurlo[a-z]{0,4} [a-z]{1,20}$|^\bmeurend[a-z]{0,4} [a-z]{1,20}$|^\brie[a-z]{0,1}kend[a-z]{0,4} [a-z]{1,20}$|^\breukge[a-z]{0,1}vend[a-z]{0,4} [a-z]{1,20}$|^\breuk[a-z]{0,1}lo[a-z]{0,4} [a-z]{1,20}$|^\breukverwekkend[a-z]{0,4} [a-z]{1,20}$|^\bruikend[a-z]{0,2} [a-z]{1,20}$|^\bruyckend[a-z]{0,4} [a-z]{1,20}$|^\bstanklo[a-z]{0,10} [a-z]{1,20}$|^\bstin[a-z]{0,1}kend[a-z]{0,4} [a-z]{1,20}$|^\bverstikkend[a-z]{0,4} [a-z]{1,20}$|^\b[a-z]{0,2}welrie[a-z]{0,1}kend[a-z]{0,4} [a-z]{1,20}$|^\bwelruikend[a-z]{0,4} [a-z]{1,20}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_adjective_is(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} is aromatiek[a-z]{0,4}$|^\b[a-z]{1,20} is aromatisch[a-z]{0,4}$|^\b[a-z]{1,20} is aromatijk[a-z]{0,4}$|^\b[a-z]{1,20} is balsamisch[a-z]{0,4}$|^\b[a-z]{1,20} is balsemend[a-z]{0,4}$|^\b[a-z]{1,20} is balsemiek[a-z]{0,4}$|^\b[a-z]{1,20} is balsemig[a-z]{0,4}$|^\b[a-z]{1,20} is balzemig[a-z]{0,4}$|^\b[a-z]{1,20} is geparfumeerd[a-z]{0,4}$|^\b[a-z]{1,20} is [a-z]{0,2}geurig[a-z]{0,4}$|^\b[a-z]{1,20} is geurlo[a-z]{0,4}$|^\b[a-z]{1,20} is meurend[a-z]{0,4}$|^\b[a-z]{1,20} is rie[a-z]{0,1}kend[a-z]{0,4}$|^\b[a-z]{1,20} is reukge[a-z]{0,1}vend[a-z]{0,4}$|^\b[a-z]{1,20} is reuk[a-z]{0,1}lo[a-z]{0,4}$|^\b[a-z]{1,20} is reukverwekkend[a-z]{0,4}$|^\b[a-z]{1,20} is ruikend[a-z]{0,2}$|^\b[a-z]{1,20} is ruyckend[a-z]{0,4}$|^\b[a-z]{1,20} is stanklo[a-z]{0,10}$|^\b[a-z]{1,20} is stin[a-z]{0,1}kend[a-z]{0,4}$|^\b[a-z]{1,20} is verstikkend[a-z]{0,4}$|^\b[a-z]{1,20} is [a-z]{0,2}welrie[a-z]{0,1}kend[a-z]{0,4}$|^\b[a-z]{1,20} is welruikend[a-z]{0,4}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_adjective_zijn(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} z[a-z]{0,1}n aromatiek[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n aromatisch[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n aromatijk[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n balsamisch[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n balsemend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n balsemiek[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n balsemig[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n balzemig[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n geparfumeerd[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n [a-z]{0,2}geurig[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n geurlo[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n meurend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n rie[a-z]{0,1}kend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n reukge[a-z]{0,1}vend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n reuk[a-z]{0,1}lo[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n reukverwekkend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n ruikend[a-z]{0,2}$|^\b[a-z]{1,20} z[a-z]{0,1}n ruyckend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n stanklo[a-z]{0,10}$|^\b[a-z]{1,20} z[a-z]{0,1}n stin[a-z]{0,1}kend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n verstikkend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n [a-z]{0,2}welrie[a-z]{0,1}kend[a-z]{0,4}$|^\b[a-z]{1,20} z[a-z]{0,1}n welruikend[a-z]{0,4}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_verb(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} \bgeurt$|^\b[a-z]{1,20} \bgeurde[a-z]{0,1}$|^\b[a-z]{1,20} \bmeurt$|^\b[a-z]{1,20} \bmeurd[a-z]{0,2}$|^\b[a-z]{1,20} \bstinck[a-z]{0,2}$|^\b[a-z]{1,20} \bstink[a-z]{0,2}$|^\b[a-z]{1,20} \brieck[a-z]{0,2}$|^\b[a-z]{1,20} \briek[a-z]{0,2}$|^\b[a-z]{1,20} \bruik[a-z]{0,2}$|^\b[a-z]{1,20} \bruyck[a-z]{0,2}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_verb_two(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} [a-z]{1,20} \bgeurt$|^\b[a-z]{1,20} [a-z]{1,20} \bgeurde[a-z]{0,1}$|^\b[a-z]{1,20} [a-z]{1,20} \bmeurt$|^\b[a-z]{1,20} [a-z]{1,20} \bmeurd[a-z]{0,2}$|^\b[a-z]{1,20} [a-z]{1,20} \bstinck[a-z]{0,2}$|^\b[a-z]{1,20} [a-z]{1,20} \bstink[a-z]{0,2}$|^\b[a-z]{1,20} [a-z]{1,20} \brieck[a-z]{0,2}$|^\b[a-z]{1,20} [a-z]{1,20} \briek[a-z]{0,2}$|^\b[a-z]{1,20} [a-z]{1,20} \bruik[a-z]{0,2}$|^\b[a-z]{1,20} [a-z]{1,20} \bruyck[a-z]{0,2}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_hare(x):
    return SOURCE if re.search(r"^\bha[a-z]{0,1}r[a-z]{0,3} (.*?)\bgeur$|^\bha[a-z]{0,1}r[a-z]{0,3} (.*?)\breuck$|^\bha[a-z]{0,1}r[a-z]{0,3} (.*?)\breuk$|^\bha[a-z]{0,1}r[a-z]{0,3} (.*?)\bstanck$|^\bha[a-z]{0,1}r[a-z]{0,3} (.*?)\bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_de_hare(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} ha[a-z]{0,1}r[a-z]{0,3} (.*?)\bgeur$|^\b[a-z]{1,20} ha[a-z]{0,1}r[a-z]{0,3} (.*?)\breuck$|^\b[a-z]{1,20} ha[a-z]{0,1}r[a-z]{0,3} (.*?)\breuk$|^\b[a-z]{1,20} ha[a-z]{0,1}r[a-z]{0,3} (.*?)\bstanck$|^\b[a-z]{1,20} ha[a-z]{0,1}r[a-z]{0,3} (.*?)\bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_zijn_other(x):
    return SOURCE if re.search(r"^\bzijn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\bgeur$|^\bzijn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\breuck$|^\bzijn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\breuk$|^\bzijn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\bstanck$|^\bzijn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\bstank$|^\bzyn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\bgeur$|^\bzyn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\breuck$|^\bzyn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\breuk$|^\bzyn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\bstanck$|^\bzyn[a-z]{0,2} [a-z]{0,20}[ ]{0,1}\bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_de_zijn(x):
    return SOURCE if re.search(r"^\b[a-z]{1,20} zijn[a-z]{0,2} (.*?)\bgeur$|^\b[a-z]{1,20} zijn[a-z]{0,2} (.*?)\breuck$|^\b[a-z]{1,20} zijn[a-z]{0,2} (.*?)\breuk$|^\b[a-z]{1,20} zijn[a-z]{0,2} (.*?)\bstanck$|^\b[a-z]{1,20} zijn[a-z]{0,2} (.*?)\bstank$|^\b[a-z]{1,20} zyn[a-z]{0,2} (.*?)\bgeur$|^\b[a-z]{1,20} zyn[a-z]{0,2} (.*?)\breuck$|^\b[a-z]{1,20} zyn[a-z]{0,2} (.*?)\breuk$|^\b[a-z]{1,20} zyn[a-z]{0,2} (.*?)\bstanck$|^\b[a-z]{1,20} zyn[a-z]{0,2} (.*?)\bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_hun(x):
    return SOURCE if re.search(r"^\bhun[a-z]{0,3} (.*?)\bgeur$|^\bhun[a-z]{0,3} (.*?)\breuck$|^\bhun[a-z]{0,3} (.*?)\breuk$|^\bhun[a-z]{0,3} (.*?)\bstanck$|^\bhun[a-z]{0,3} (.*?)\bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_source_de_hun(x):
    return SOURCE if re.search(r"^\ba-z]{1,20} hun[a-z]{0,3} (.*?)\bgeur$|^\b[a-z]{1,20} hun[a-z]{0,3} (.*?)\breuck$|^\b[a-z]{1,20} hun[a-z]{0,3} (.*?)\breuk$|^\b[a-z]{1,20} hun[a-z]{0,3} (.*?)\bstanck$|^\ba-z]{1,20} hun[a-z]{0,3} (.*?)\bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_evoked_naar(x):
    return EVOKED if re.search(r"^\bgeur[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|^\breuck[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|^\breuk[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|^\brieck[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|^\briek[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|$^\bstanck[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|$^\bstank[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|$^\bstinck[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|$^\bstink[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|^\bruik[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}$|^\bruyck[a-z]{0,5} na[a-z]{0,2} [a-z]{1,20}", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_evoked_als(x):
    return EVOKED if re.search(r"^\bgeur[a-z]{0,5} als (.*?)$|^\breuck[a-z]{0,5} als (.*?)$|^\breuk[a-z]{0,5} als (.*?)$|^\brieck[a-z]{0,5} als (.*?)$|^\briek[a-z]{0,5} als (.*?)$|$^\bstanck[a-z]{0,5} als (.*?)$|$^\bstank[a-z]{0,5} als (.*?)$|$^\bstinck[a-z]{0,5} als (.*?)$|$^\bstink[a-z]{0,5} als (.*?)$|^\bruik[a-z]{0,5} als (.*?)$|^\bruyck[a-z]{0,5} als (.*?)", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_evoked_als_other(x):
    return EVOKED if re.search(r"^\bgeur[a-z]{0,5} [a-z]{1,20} als (.*?)$|^\breuck[a-z]{0,5} [a-z]{1,20} als (.*?)$|^\breuk[a-z]{0,5} [a-z]{1,20} als (.*?)$|^\brieck[a-z]{0,5} [a-z]{1,20} als (.*?)$|^\briek[a-z]{0,5} [a-z]{1,20} als (.*?)$|$^\bstanck[a-z]{0,5} [a-z]{1,20} als (.*?)$|$^\bstank[a-z]{0,5} [a-z]{1,20} als (.*?)$|$^\bstinck[a-z]{0,5} [a-z]{1,20} als (.*?)$|$^\bstink[a-z]{0,5} [a-z]{1,20} als (.*?)$|^\bruik[a-z]{0,5} [a-z]{1,20} als (.*?)$|^\bruyck[a-z]{0,5} [a-z]{1,20} als (.*?)", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_quality_keyword(x):
    return QUALITY if re.search(r"^\baromatiek[a-z]{0,4}$|^\baromatisch[a-z]{0,4}$|^\baromatijk[a-z]{0,4}$|^\bbalsamisch[a-z]{0,4}$|^\bbalsemend[a-z]{0,4}$|^\bbalsemiek[a-z]{0,4}$|^\bgeparfumeerd[a-z]{0,4}$|^\bgeurig[a-z]{0,4}$|^\bgeurlo[a-z]{0,4}$|^\bmeurend[a-z]{0,4}$|^\bongeurig[a-z]{0,4}$|^\bonriekba[a-z]{0,4}$|^\bonwelrieckend[a-z]{0,4}$|^\bonwelriekend[a-z]{0,4}$|^\b[a-z]{rieckend[a-z]{0,4}$|^\b[a-z]{0,4}riekend[a-z]{0,4}$|^\breukgeevend[a-z]{0,4}$|^\breukgevend[a-z]{0,4}$|^\breukelo[a-z]{0,4}$|^\breuklo[a-z]{0,4}$|^\breukverwekkend[a-z]{0,4}$|^\bruikend[a-z]{0,2}$|^\bruykend[a-z]{0,4}$|^\bstankverdryvend[a-z]{0,4}$|^\bstanklo[a-z]{0,10}$|^\bstankverdrijvend[a-z]{0,4}$|^\bstankverwerend[a-z]{0,4}$|^\bstinckend[a-z]{0,4}$|^\bstinkend[a-z]{0,4}$|^\bverstikkend[a-z]{0,4}$|^\bwelrieckend[a-z]{0,4}$|^\bwelriekend[a-z]{0,4}$|^\bwelruikend[a-z]{0,4}$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_quality_een(x):
    return QUALITY if re.search(r"^\b[a-z]{1,20} \bgeur$|^\b[a-z]{1,20} \bgeur$|^\b[a-z]{1,20} \breuck$|^\b[a-z]{1,20} \breuck$|^\b[a-z]{1,20} \bstanck$|^\b[a-z]{1,20} \bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_quality_een_two(x):
    return QUALITY if re.search(r"^\b[a-z]{1,20} [a-z]{1,20} \bgeur$|^\b[a-z]{1,20} [a-z]{1,20} \bgeur$|^\b[a-z]{1,20} [a-z]{1,20} \breuck$|^\b[a-z]{1,20} [a-z]{1,20} \breuck$|^\b[a-z]{1,20} [a-z]{1,20} \bstanck$|^\b[a-z]{1,20} [a-z]{1,20} \bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_quality_en(x):
    return QUALITY if re.search(r"^[a-z]{1,20} en (.*?) \bgeur$|^[a-z]{1,20} en (.*?) \breuck$|^a-z]{1,20} en (.*?) \breuk$|^a-z]{1,20} en (.*?) \bstanck$|^a-z]{1,20} en (.*?) \bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_quality_van(x):
    return QUALITY if re.search(r"^(.*?) van \bgeur$|^(.*?) van \breuck$|^(.*?) van \breuk$|^(.*?) van \bstanck$|^(.*?) van \bstank$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_quality_is(x):
    return QUALITY if re.search(r"^\bgeur (.*?) is$|^\breuck (.*?) is$|^\breuk (.*?) is$|^\bstanck (.*?) is$|^\bstank (.*?) is$", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_quality_test(x):
    return QUALITY if re.search(r"^[a-z]{1,20} [a-z]{1,20} [a-z]{1,20} \bgeur$|^\b[a-z]{1,20} [a-z]{1,20} [a-z]{1,20} \bgeur$|^\b[a-z]{1,20} [a-z]{1,20} [a-z]{1,20} \breuck$|^\b[a-z]{1,20} [a-z]{1,20} [a-z]{1,20} \breuck$|^\b[a-z]{1,20} [a-z]{1,20} [a-z]{1,20} \bstanck$|^\b[a-z]{1,20} [a-z]{1,20} [a-z]{1,20} \bstank$", x.word, flags=re.I) else NO_SMELL

# Define set of labelling functions
lfs = [
    # Source
    lf_regex_source_keyword,
    lf_regex_source_van,
    lf_regex_source_van_two,
    lf_regex_source_van_three,
    lf_regex_source_heeft,
    lf_regex_source_geeft,
    lf_regex_source_heeft_geeft,
    lf_regex_source_heeft_geeft_two,
    lf_regex_source_is,
    lf_regex_source_zijn,
    lf_regex_source_is_zijn,
    lf_regex_source_dezer,
    lf_regex_source_der,
    lf_regex_source_der_two,
    lf_regex_source_welke,
    lf_regex_source_welke_two,
    lf_regex_source_aan,
    lf_regex_source_adjective,
    lf_regex_source_adjective_is,
    lf_regex_source_adjective_zijn,
    lf_regex_source_verb,
    lf_regex_source_verb_two,
    lf_regex_source_hare,
    lf_regex_source_de_hare,
    lf_regex_source_zijn_other,
    lf_regex_source_de_zijn,
    lf_regex_source_hun,
    lf_regex_source_de_hun,
    # Evoked
    lf_regex_evoked_naar,
    lf_regex_evoked_als,
    lf_regex_evoked_als_other,
    # Quality
    lf_regex_quality_keyword,
    lf_regex_quality_een,
    lf_regex_quality_een_two,
    lf_regex_quality_en,
    lf_regex_quality_van,
    lf_regex_quality_is
]

# Apply labelling functions to unlabelled training data
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

# Get summary of labelling functions
LFAnalysis(L=L_train, lfs=lfs).lf_summary()

y_test_int = Y_test.astype('int32')

# Get summary of labelling functions
LFAnalysis(L=L_test, lfs=lfs).lf_summary(y_test_int)

# Create majority voting model
majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)

# Train label model and compute training labels
label_model = LabelModel(cardinality=4, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50)
df_train["label"] = label_model.predict(L=L_train)

# Calculate accuracy of majority voting model
majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

# Calculate accuracy of labelling model
label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

# Filter out no smell
df_train = df_train[df_train.label != NO_SMELL]

# Show dataframe
df_train.head()

# Filter on evoked label
df_source = df_train.loc[df_train['label'] == 1]

# Replace substrings and split words
# Apply to: Van, van_two, van_three, dezer, der, der_two, adjective, aan
df_source['word'] = df_source['word'].str.replace(r'(.*?) ', '')

# Apply to: Heeft, geeft, is, zijn, welke, adjective_is, adjective_zijn, verb, verb_two, de_hare, zijn_other, de_zijn, hun
df_source['word'] = df_source['word'].str.split(' ').str[0]

# Apply to: Heeft_geeft, is_zijn
df_source['word'] = df_source['word'].str.split(' ').str[1]

# Apply to: Heeft_geeft_two
df_source['word'] = df_source['word'].str.replace(r'^[a-z]{1,20} ', '')
df_source['word'] = df_source['word'].str.replace(r' [a-z]{1,20}$', '')

# Apply to: Heeft_geeft_two, is_zijn
df_source = df_source[~df_source.word.str.contains(r"\bgeur|\breuck|\breuk|\bstanck|\bstanck")]

# Apply to: Heeft_geeft_two
df_source = df_source.assign(word=df_source['word'].str.split(' ')).explode('word')

df_source['word'] = df_source['word'].str.replace(r' ', '')

df_source.head()

# Filter on evoked label
df_evoked = df_train.loc[df_train['label'] == 2]

# Replace substrings and split words
# Apply to: Naar
df_evoked['word'] = df_evoked['word'].str.replace(r'(.*?) ', '')

# Apply to: Als
df_evoked['word'] = df_evoked['word'].str.replace(r'^[a-z]{1,20} [a-z]{1,20} ', '')

# Apply to: Als_other
df_evoked['word'] = df_evoked['word'].str.replace(r'^[a-z]{1,20} [a-z]{1,20} ', '')

# Apply to: Als
df_evoked = df_evoked.assign(word=df_evoked['word'].str.split(' ')).explode('word')

df_evoked['word'] = df_evoked['word'].str.replace(r' ', '')

df_evoked

# Filter on quality label
df_quality = df_train.loc[df_train['label'] == 3]

# Replace substrings and split words
# Apply to: Een
df_quality['word'] = df_quality['word'].str.split(' ').str[0]

# Apply to: Een_two
df_quality['word'] = df_quality['word'].loc[(df_pos['word'] == "zeer")

# Apply to: En, van
df_quality['word'] = df_quality['word'].str.replace(r' [a-z]{1,20}$', '')

# Apply to: En
df_quality['word'] = df_quality['word'].str.replace(r'\ben\b', '')

# Apply to: Van
df_quality['word'] = df_quality['word'].str.replace(r'\bvan\b', '')

# Apply to: Van, is
df_quality['word'] = df_quality['word'].str.replace(r'\bis\b', '')

# Apply to: Is
df_quality['word'] = df_quality['word'].str.replace(r'^[a-z]{1,20} ', '')

# Apply to: En, van
df_quality = df_quality.assign(word=df_quality['word'].str.split(' ')).explode('word')

df_quality['word'] = df_quality['word'].str.replace(r' ', '')

df_quality.head()

# Append dataframes
df_train = df_source.append([df_evoked, df_quality])

# Show dataframe
df_train.head()

# Save dataframe
df_train.to_excel("output_lf_sr_word.xlsx", engine='xlsxwriter')

# Load Dutch Spacy set
nlp = spacy.load('nl_core_news_sm')

# Define new dataframe
df_pos = df_train

# Create part-of-speech-tag for output words
pos = []

for doc in nlp.pipe(df_pos['word'].astype('unicode').values, batch_size=50,
                        n_threads=3):
    if doc.is_parsed:
        pos.append([n.pos_ for n in doc])
    else:
        pos.append(None)

df_pos['pos'] = pos

df_pos['pos'] = df_pos['pos'].astype(str)

# Show dataframe
df_pos.head()

# Apply to: Van
df_pos = df_pos.loc[(df_pos['pos'] == "['NOUN']") | (df_pos['pos'] == "['ADJ']") | (df_pos['word'] == "zich") & (df_pos['label'] == 1)]

# Apply to: Van_two, van_three, heeft_geeft_two, der, welke_two, verb
df_pos = df_pos.loc[(df_pos['pos'] == "['NOUN']") | (df_pos['pos'] == "['ADJ']") & (df_pos['label'] == 1)]

# Apply to: Heeft, geeft, heeft_geeft, is, is_zijn, dezer, der_two, welke, adjective, aan, verb_two, de_hare
df_pos = df_pos.loc[(df_pos['pos'] == "['NOUN']") & (df_pos['label'] == 1)]

# Apply to: Zijn, adjective_zijn
df_pos = df_pos.loc[(df_pos['pos'] == "['NOUN']") | (df_pos['word'] == "zij") | (df_pos['word'] == "zy") & (df_pos['label'] == 1)]

# Apply to: Adjective_is, adjective_zijn de_zijn, de_hun
df_pos = df_pos.loc[(df_pos['pos'] == "['NOUN']") | (df_pos['word'] == "het") & (df_pos['label'] == 1)]

# Apply to: Naar, als
df_pos = df_pos.loc[(df_pos['pos'] == "['NOUN']") | (df_pos['pos'] == "['ADJ']") & (df_pos['label'] == 2)]

# Apply to: Een, en, van
df_pos = df_pos.loc[(df_pos['pos'] == "['NOUN']") | (df_pos['pos'] == "['ADJ']") | (df_pos['pos'] == "['VERB']") & (df_pos['label'] == 3)]

# Apply to: Is
df_pos = df_pos.loc[(df_pos['pos'] == "['ADJ']") & (df_pos['label'] == 3)]

# Show dataframe
df_pos.head() 

# Save dataframe
df_pos.to_excel("output_lf_sr_pos.xlsx", engine='xlsxwriter')

L_test = np.argmax(L_test,axis=1)

# Calculate micro-averaged precision, recall, and F1 for majority label
precision_recall_fscore_support(Y_test, L_test, labels=[1])

# Calculate micro-averaged precision, recall, and F1
precision_recall_fscore_support(Y_test, L_test, average='micro', labels=[1, 2, 3])

# Calculate accuracy
accuracy_score(Y_test, L_test)

