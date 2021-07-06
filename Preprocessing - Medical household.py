#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize


# In[2]:


# Open and read file
with open('/Users/melissavandenbovenkamp/Documents/Master Data Science/Scriptie/Data/Medisch huishoudelijk/VanderHaar-1788-Proeve over de herzenen en zenuwen, en eenige derzelver ziekten.txt', 'r') as file:
    data = file.read()


# In[3]:


# Remove text breaks consiting of a letter, space, and number
data = data.replace(r"[A-Z]{1} [0-9]{1}", "")

# Remove headers after tabs, command this when having texts with page numbers
data = re.sub("\n\n(.*?)\n\n(.*?)\n", "\n", data)

# Remove page numbers and image texts
data = re.sub("{>>pagina-aanduiding<<}", "", data)
data = re.sub("{==(.*?)==}", "", data)
data = re.sub("{== afbeelding", "", data)
data = re.sub("==} {>>afbeelding<<}", "", data)

# Delete double words
data = re.sub("\b(\w+)( \1\b)+", "\1", data)

# Remove footnotes starting with something between brackets e.g. (*7)
data = re.sub("\n[(](.*?)[)](.*?)\n", "", data)

# Remove tabs
data = data.replace(r"\n", " ")


# In[4]:


# Tokenize text into sentences
tokenized_text=sent_tokenize(data)
print(tokenized_text)


# In[5]:


# Put tokenized text into dataframe
df = pd.DataFrame(tokenized_text)

# Show dataframe
df


# In[6]:


# Remove rows with numbers
df = df[~df[0].str.contains(r'[0-9]')]

# Remove rows with reference to page
df = df[~df[0].str.contains(r'bl.')]
df = df[~df[0].str.contains(r'bladz.')]
df = df[~df[0].str.contains(r'pag.')]

# Remove rows with special characters
df = df[~df[0].str.contains(r'[@#$%+/*><^=]')]

# Count number of words and remove rows with less than 5 words
count = df[0].str.split().str.len()
df = df[~(count<5)].copy()

# Show dataframe
df


# In[7]:


# Save dataframe
df.to_excel("output.xlsx")


# In[ ]:




