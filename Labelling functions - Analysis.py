# Import packages
import pandas as pd
import re
import numpy as np

# Import datasets
df = pd.read_excel()

# Filter on word column
df = df[['word']]

# Group by word and count number of occurences
df['count'] = df.groupby('word')['word'].transform('count')

# Drop duplicates
df =  df.drop_duplicates()

#Assigne keyword to words
df['keyword'] = pd.np.where(df.word.str.contains("aroma"), "aroma",
                        pd.np.where(df.word.str.contains("aspiratie"), "aspiratie",
                        pd.np.where(df.word.str.contains("balsem"), "balsem",
                        pd.np.where(df.word.str.contains("balzem"), "balzem",
                        pd.np.where(df.word.str.contains("bisam"), "bisam",
                        pd.np.where(df.word.str.contains("civet"), "civet",
                        pd.np.where(df.word.str.contains("damp"), "damp",
                        pd.np.where(df.word.str.contains("eau-de-"), "eau-de-",
                        pd.np.where(df.word.str.contains("geur"), "geur",
                        pd.np.where(df.word.str.contains("gommen"), "gommen",
                        pd.np.where(df.word.str.contains("inadem"), "inadem",
                        pd.np.where(df.word.str.contains("loddereindoos"), "loddereindoos",
                        pd.np.where(df.word.str.contains("lodereindoos"), "lodereindoos",
                        pd.np.where(df.word.str.contains("luchtje"), "luchtje",
                        pd.np.where(df.word.str.contains("miasma"), "miasma",
                        pd.np.where(df.word.str.contains("mirre"), "mirre",
                        pd.np.where(df.word.str.contains("mirrhe"), "mirrhe",
                        pd.np.where(df.word.str.contains("mirthe"), "mirthe",
                        pd.np.where(df.word.str.contains("myrre"), "myrre",
                        pd.np.where(df.word.str.contains("myrrhe"), "myrrhe",
                        pd.np.where(df.word.str.contains("muscus"), "muscus",
                        pd.np.where(df.word.str.contains("muskus"), "muskus",
                        pd.np.where(df.word.str.contains("nardus"), "nardus",
                        pd.np.where(df.word.str.contains("odeur"), "odeur",
                        pd.np.where(df.word.str.contains("ophaling"), "ophaling",
                        pd.np.where(df.word.str.contains("parfuim"), "parfuim",
                        pd.np.where(df.word.str.contains("parfum"), "parfum",
                        pd.np.where(df.word.str.contains("parfuym"), "parfuym",
                        pd.np.where(df.word.str.contains("peerfum"), "peerfum",
                        pd.np.where(df.word.str.contains("perfum"), "perfum",
                        pd.np.where(df.word.str.contains("pomander"), "pomander",
                        pd.np.where(df.word.str.contains("smook"), "smook",
                        pd.np.where(df.word.str.contains("smoor"), "smoor",
                        pd.np.where(df.word.str.contains("snuif"), "snuif",
                        pd.np.where(df.word.str.contains("stanck"), "stanck",
                        pd.np.where(df.word.str.contains("stank"), "stank",
                        pd.np.where(df.word.str.contains("stinck"), "stinck",
                        pd.np.where(df.word.str.contains("stink"), "stink",
                        pd.np.where(df.word.str.contains("stoom"), "stoom",
                        pd.np.where(df.word.str.contains("waassem"), "waassem",
                        pd.np.where(df.word.str.contains("waazem"), "waazem",
                        pd.np.where(df.word.str.contains("wadem"), "wadem",
                        pd.np.where(df.word.str.contains("waesem"), "waesem",
                        pd.np.where(df.word.str.contains("walm"), "walm",
                        pd.np.where(df.word.str.contains("wasem"), "wasem",
                        pd.np.where(df.word.str.contains("wieroken"), "wieroken",
                        pd.np.where(df.word.str.contains("wierook"), "wierook",
                        pd.np.where(df.word.str.contains("wijrook"), "wijrook",
                        pd.np.where(df.word.str.contains("wyroock"), "wyroock",
                        pd.np.where(df.word.str.contains("zalf"), "zalf",
                        pd.np.where(df.word.str.contains("gemeurd"), "gemeurd",
                        pd.np.where(df.word.str.contains("geroken"), "geroken",
                        pd.np.where(df.word.str.contains("smoken"), "smoken",
                        pd.np.where(df.word.str.contains("smoren"), "smoren",
                        pd.np.where(df.word.str.contains("inzuigen"), "inzuigen",
                        pd.np.where(df.word.str.contains("ophaling"), "ophaling",
                        pd.np.where(df.word.str.contains("ronfelen"), "ronfelen",
                        pd.np.where(df.word.str.contains("snoef"), "snoef",
                        pd.np.where(df.word.str.contains("snoof"), "snoof",
                        pd.np.where(df.word.str.contains("snoven"), "snoven",
                        pd.np.where(df.word.str.contains("snuiven"), "snuiven",
                        pd.np.where(df.word.str.contains("stonk"), "stonk",
                        pd.np.where(df.word.str.contains("verlucht"), "verlucht",
                        pd.np.where(df.word.str.contains("balsamisch"), "balsamisch",
                        pd.np.where(df.word.str.contains("vermuf"), "vermuf",
                        pd.np.where(df.word.str.contains("verstikkend"), "verstikkend",
                        pd.np.where(df.word.str.contains("welrieckend"), "welrieckend",
                        pd.np.where(df.word.str.contains("welriekend"), "welriekend",
                                    
                        pd.np.where(df.word.str.contains(r"\baromen\b"), "aromen",
                        pd.np.where(df.word.str.contains(r"\baroom\b"), "aroom",
                        pd.np.where(df.word.str.contains(r"\bgom\b"), "gom",
                        pd.np.where(df.word.str.contains(r"\bmeur\b"), "meur",
                        pd.np.where(df.word.str.contains(r"\bmeuren\b"), "meuren",
                        pd.np.where(df.word.str.contains(r"\bmuf\b"), "muf",
                        pd.np.where(df.word.str.contains(r"\bneus\b"), "neus",
                        pd.np.where(df.word.str.contains(r"\bneuz\b"), "neuz",
                        pd.np.where(df.word.str.contains(r"\bneuze\b"), "neuze",
                        pd.np.where(df.word.str.contains(r"\bneuzen\b"), "neuzen",
                        pd.np.where(df.word.str.contains(r"\bodor\b"), "odor",
                        pd.np.where(df.word.str.contains(r"\bodorant\b"), "odorant",
                        pd.np.where(df.word.str.contains(r"\breeuw\b"), "reeuw",
                        pd.np.where(df.word.str.contains(r"\broken\b"), "roken",
                        pd.np.where(df.word.str.contains(r"\brook\b"), "rook",
                        pd.np.where(df.word.str.contains(r"\bseef\b"), "seef",
                        pd.np.where(df.word.str.contains(r"\bzaan\b"), "zaan",
                                    
                        pd.np.where(df.word.str.contains(r"\breuck"), "reuck",
                        pd.np.where(df.word.str.contains(r"\breuk"), "reuk",
                        pd.np.where(df.word.str.contains(r"\brieck"), "rieck",
                        pd.np.where(df.word.str.contains(r"\briek"), "riek",
                        pd.np.where(df.word.str.contains(r"\bruik"), "ruik",
                        pd.np.where(df.word.str.contains(r"\bruyck"), "ruyck",
                        pd.np.where(df.word.str.contains(r"\nsnuf"), "snuf",
                                    
                        pd.np.where(df.word.str.contains(r"olie\b"), "olie",
                        pd.np.where(df.word.str.contains(r"oliën\b"), "oliën",
                        pd.np.where(df.word.str.contains(r"oly\b"), "oly", "No keyword")))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

# Show dataframe
df.head()

# Save dataframe
df.to_excel("output_lf_analyses_variations.xlsx", engine='xlsxwriter')

# Group by kewyord and count total number of occurences
df_count_total = df.groupby("keyword")["count"].sum()

# Transform series to frame
df_count_total.to_frame()

# Create new dataframe with keyword column
df_count_keyword = df[['keyword']].copy()

# Group by keyword and count word variations
df_count_keyword['count'] = df_count_keyword.groupby('keyword')['keyword'].transform('count')

# Drop duplicates
df_count_keyword = df_count_keyword.drop_duplicates()

# Join count dataframes
df_count = df_count_keyword.merge(df_count_total,on='keyword',how='left')

# Show dataframe
df_count.head()

# Save dataframe
df_count.to_excel("output_lf_analyses_count.xlsx", engine='xlsxwriter')

# Combine all words per keyword
df_combined = df[['keyword', 'word']].copy()

df_combined = df_combined.sort_values(by=["keyword", "word"])
df_combined = df_combined.groupby("keyword")

df_combined = df_combined["word"].agg(lambda column: ", ".join(column))

df_combined = df_combined.reset_index(name="words")

# Show dataframe
df_combined.head()

# Save dataframe
df_combined.to_excel("output_lf_analyses_words.xlsx", engine='xlsxwriter')

