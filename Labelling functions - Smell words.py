# Import packages
import pandas as pd
import re
import numpy as np

from snorkel.labeling import LabelingFunction
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelModel
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import filter_unlabeled_dataframe

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Import datasets
df = pd.read_excel()
df2 = pd.read_excel()

# Define new dataframes
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

# Concatenate all series vertically
df_train = pd.concat([unigrams]).dropna().reset_index(drop=False)

# Rename columns
df_train.rename(columns = {'level_0':'row', 0:'word'}, inplace=True)

# Show dataframe
df_train

# Merge dataframes on row numbers
df_train = pd.merge(df_train[['row', 'word']],
           df[['row', 'text', 'source']],
           on = 'row',
           how = 'left')

# Define labels
NO_SMELL = 0
SMELL = 1

# Define keyword functions
def keyword_lookup(x, keywords, label):
    if any(word in x.word.lower() for word in keywords):
        return label
    return NO_SMELL

def make_keyword_lf(keywords, label=SMELL):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

keyword_smell = make_keyword_lf(keywords=["aroma", "aspiratie", "balsem", "balzem", "bisam", "civet", "damp", "eau-de-", "geur", "gommen", "inadem", "lodderein", "loderein", "luchtje", "mirre", "mirrhe", "myrre", "myrrhe", "mirthe", "muscus", "muskus", "odeur", "ophaling", "parfuim", "parfum", "parfuym", "perfum", "perfuum", "pomander", "smook", "smoor", "snuif", "stancnk", "stank", "stinck", "stink", "stoom", "waassem", "waazem", "wadem", "waesem", "walm", "wasem", "wieroken", "wierook", "wijrook", "wyroock", "zalf", "smoken", "smoren", "gemeurd", "geroken", "inzuigen", "ophaling", "snoef", "snoof", "snoven", "snuiven", "stonk", "verlucht", "balsamisch", "vermuf", "verstikkend", "welrieckend", "welriekend"])

@labeling_function()
def keyword_smell_other(x):
    keywords = ["aromen", "aroom", "gom", "meur", "meuren", "miasma", "muf", "neus", "neuz", "neuze", "neuzen", "odor", "odorant", "reeuw", "roken", "rook", "seef", "zaan"]
    for word in x.word.lower().split():
        if word in keywords:
            return SMELL
    return NO_SMELL

# Define regular expressions
@labeling_function()
def lf_regex_start(x):
    return SMELL if re.search(r"\bnardus|\breuk|\breuck|\briek|\brieck|\bruik|\bruyck|\snuf", x.word, flags=re.I) else NO_SMELL

@labeling_function()
def lf_regex_end(x):
    return SMELL if re.search(r"olie\b|oliën\b|oly\b", x.word, flags=re.I) else NO_SMELL

# Define set of labelling functions
lfs = [
    keyword_smell,
    keyword_smell_other,
    lf_regex_start,
    lf_regex_end,
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

# Train labelling model and compute training labels
label_model = LabelModel(cardinality=2, verbose=True)
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

# Save dataframe
df_train.to_excel("output_lf_sw_word.xlsx", engine='xlsxwriter')

L_test = np.argmax(L_test,axis=1)

# Calculate micro-averaged precision, recall, and F1
precision_recall_fscore_support(Y_test, L_test, average='micro')

# Calculate accuracy
accuracy_score(Y_test, L_test)



