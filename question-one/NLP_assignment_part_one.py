#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This notebook was developed in Google Colab with files stored in Google Drive.


# In[ ]:


#each part is submitted as separate .py files


# In[ ]:


#git logs included as text files, pickles folder excluded
#due to file size constraints


# In[ ]:


import os
import random

import nltk
import spacy
from pathlib import Path

import math
import re
from collections import Counter
import pandas as pd
from spacy.tokens import Doc

#detect environment and set project path automatically
try:
    #Colab - code developed by student here
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/cw-pack-2026'

except ImportError:
    #running locally - set to unzip file location
    PROJECT = str(Path(__file__).parent) if '__file__' in dir() else str(Path.cwd())

os.chdir(PROJECT)
print("Working in:", PROJECT)


# In[ ]:


#git commit template
get_ipython().system('git add .')
get_ipython().system('git commit -m "Set up example"')


# In[ ]:


import os
os.makedirs("content/drive/MyDrive/cw-pack-2026/texts/novels", exist_ok=True)

from google.colab import files
uploaded = files.upload()

for filename in uploaded:
  os.rename(filename, f"texts/novels/{filename}")


# In[ ]:


#git commit
get_ipython().system('git add .')
get_ipython().system('git commit -m "File upload"')


# In[ ]:


get_ipython().system('pip install spacy --quiet')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[ ]:


#NLP assignment
#due 26 June 2pm

#not all novels in the texts file are from the 19th century but are included in the script
#to exclude novels published after 1900 ->
  #df = df[df["year"] < 1900]

#part 1
#follows template provided

#load spaCy and make maximum text length larger as some texts exceed spaCy's maximum length as mentioned in d) iv)
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


# In[ ]:


#git commit
get_ipython().system('git add .')
get_ipython().system('git commit -m "imports"')


# In[ ]:


#c) FK grade level
def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """

    #note: template says FK grade level but brief says ease score of reading

    #tokenize into sentences and alphabetic words only
    sentences = nltk.sent_tokenize(text)
    words = [w for w in nltk.word_tokenize(text) if w.isalpha()]

    num_sentences = len(sentences)
    num_words = len(words)

    #guard against empty texts
    if num_sentences == 0 or num_words == 0:
        return 0.0

    #count syllables across all words using count_syl
    num_syllables = sum(count_syl(w, d) for w in words)

    #average sentence length
    asl = num_words / num_sentences

    #average syllables per word
    asw = num_syllables / num_words

    #FK grade level formula
    return 0.39 * asl + 11.8 * asw - 15.59


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "define FK level function for part c"')


# In[ ]:


#c) helper: count syllables in a single word
def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """

    word_lower = word.lower()
    if word_lower in d:
        #phonemes ending with a digit carry stress -> mark one syllable
        #eg 'AH0', 'EY1' are vowel sounds but 'T', 'K' etc. are not
        return sum(1 for phoneme in d[word_lower][0] if phoneme[-1].isdigit())
    else:
        #fallback for words not in CMU dictionary:
        #count contiguous vowel groups as one syllable each
        clusters = re.findall(r"[aeiouAEIOU]+", word)
        count = len(clusters)

        #discount silent trailing 'e' eg "make", "love"
        if word_lower.endswith("e") and count > 1:
            count -= 1
        #every word has minimum one syllable
        return max(1, count)


# In[ ]:


#git commit
get_ipython().system('git add .')
get_ipython().system('git commit -m "define helper syllable counting function"')


# In[ ]:


#a) read novels from directory into DataFrame
def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""

    records = []
    for filepath in sorted(path.glob("*.txt")):
        #parse data from filename eg "Dracula-Stoker-1897"
        stem  = filepath.stem
        parts = stem.split("-")
        year   = re.search(r"\d{4}", parts[-1]).group() #four-digit year only, removes (1) eg
        author = parts[-2].strip()
        #everything preceding last two parts is the title
        #deals with titles containing hyphens
        title  = "-".join(parts[:-2]).strip()

        #read novel text and replace any undecodable bytes
        text = filepath.read_text(encoding="utf-8", errors="replace")
        records.append({
            "text":   text,
            "title":  title,
            "author": author,
            "year":   int(year),
        })

    df = pd.DataFrame(records, columns=["text", "title", "author", "year"])
    #sort chronologically to follow order of publication
    df = df.sort_values("year").reset_index(drop=True)
    return df


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "read novel function to load texts from directory"')


# In[ ]:


#d) parse texts with spaCy and pickle results
def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes
    the resulting  DataFrame to a pickle file"""

    store_path.mkdir(parents=True, exist_ok=True)

    docs = []
    for i, (_, row) in enumerate(df.iterrows()):
      print(f"Parsing {row['title']} ({i + 1} / {len(df)})...")
      text = row["text"]
      if len(text) > nlp.max_length:
        #split oversized texts into max_length chunks, parse, and concatenate into single Doc using Doc.from_docs below
        chunks = [text[i:i + nlp.max_length]
                  for i in range(0, len(text), nlp.max_length)]
        combined = None
        for chunk in chunks:
            chunk_doc = nlp(chunk)
            if combined is None:
                combined = chunk_doc
            else:
                combined = Doc.from_docs([combined, chunk_doc])
        docs.append(combined)
      else:
        docs.append(nlp(text))

    df = df.copy()
    df["doc"] = docs

    #convert to pickle to avoid re-parsing on every run
    pickle_path = store_path / out_name
    df.to_pickle(pickle_path)
    df = pd.read_pickle(pickle_path)
    print(f"\nParsed DataFrame saved to {pickle_path}")
    return pd.read_pickle(pickle_path)


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "define parsing function to process and pickle dataframe"')


# In[ ]:


#b) type-token ratio
def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    #case-insensitive and remove punctuation
    tokens = [t.lower() for t in nltk.word_tokenize(text) if t.isalpha()]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "define nltk_ttr for type token ratio"')


# In[ ]:


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "define get_ttrs for type token ratio"')


# In[ ]:


def get_fks(df):
    """helper function to add fks to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "define get_fks to calculate FK across dataframe"')


# In[ ]:


#functions for e)
#subject and verb analysis using parsed docs

subject_labels = {"nsubj", "nsubjpass"}

def get_top_subjects(doc, n=10):
    """(e-i) Returns the n most common syntactic subjects in a spaCy Doc.

    Returns:
        list[tuple]: (lemma, count) pairs, most common first.
    """

    subjects = [
        token.lemma_.lower()
        for token in doc
        if token.dep_ in subject_labels and token.is_alpha
    ]
    return Counter(subjects).most_common(n)


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "define top_subjects to return syntactic subjects"')


# In[ ]:


def get_subject_verb_pmi(doc, subject_lemma, min_count=2):
    """(e-ii/iii) Returns verbs most likely to co-occur with a given subject, ordered by PMI.

    Pointwise Mutual Information measures how much more often a subject-verb
    pair appears together than expected by chance:

        PMI(subj, verb) = log2( P(subj, verb) / (P(subj) * P(verb)) )

    probabilities are estimated from all (subject, verb) pairs extracted via
    nsubj dependency arcs across the document.

    Args:
        doc: spaCy Doc object.
        subject_lemma (str): The subject lemma to query (e.g. 'he', 'she').

        min_count (int): Minimum co-occurrence count to include a verb.
                         Exposed as a parameter so callers can adjust it.

    Returns:
        list[tuple]: (verb_lemma, pmi) pairs sorted by PMI descending.
    """

    subject_lemma = subject_lemma.lower()

    #collect subject and verb pairs from nsubj dependency arcs
    all_pairs = []
    for token in doc:
        if token.dep_ in subject_labels and token.is_alpha:
            verb = token.head
            #and verb.is_alpha prevents apostrophe conjunctions from being included
            if verb.pos_ == "VERB" and verb.is_alpha:
                all_pairs.append((token.lemma_.lower(), verb.lemma_.lower()))

    if not all_pairs:
        return []

    total_pairs = len(all_pairs)

    #count joint and marginal frequencies
    pair_counts = Counter(all_pairs)
    subj_counts = Counter(s for s, _ in all_pairs)
    verb_counts = Counter(v for _, v in all_pairs)

    #filter pairs involving target subject and meeting min_count
    target_pairs = {v: c for (s, v), c in pair_counts.items()
                    if s == subject_lemma and c >= min_count and verb_counts[v] >= min_count * 2}

    if not target_pairs:
        return []

    #compute PMI for co-occurring verb
    pmi_scores = {}
    p_subj = subj_counts[subject_lemma] / total_pairs
    for verb, count in target_pairs.items():
        V = len(verb_counts)
        p_verb      = (verb_counts[verb] + 1) / (total_pairs + V)
        p_subj_verb = (count + 1) / (total_pairs + V)
        if p_subj > 0 and p_verb > 0:
            pmi_scores[verb] = math.log2(p_subj_verb / (p_subj * p_verb))

    #return sorted highest PMI first
    #does not filter negative PMI out
    return sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "define subject verb function to return most co-occurring verb and filter apostrophe conjunctions"')


# In[ ]:


def print_top_subjects(df, n=10):
    """(e) i) Prints the top n syntactic subjects for each novel."""

    print("\nTop Syntactic Subjects:")
    for _, row in df.iterrows():
        subjects = get_top_subjects(row["doc"], n=n)
        print(f"\n{row['title'].replace('_', ' ')}:")
        for lemma, count in subjects:
            print(f"  {lemma}: {count}")

def print_he_verbs(df):
    """(e) ii) Prints verbs most likely to co-occur with 'he' by PMI."""

    print("\nVerbs for subject 'he' (by PMI):")
    for _, row in df.iterrows():
        verbs = get_subject_verb_pmi(row["doc"], "he")
        print(f"\n{row['title'].replace('_', ' ')}:")
        for verb, pmi in verbs[:20]:
            print(f"  {verb}: {pmi:.4f}")

def print_she_verbs(df):
    """(e) iii) Prints verbs most likely to co-occur with 'she' by PMI."""

    print("\nVerbs for subject 'she' (by PMI):")
    for _, row in df.iterrows():
        verbs = get_subject_verb_pmi(row["doc"], "she")
        print(f"\n{row['title'].replace('_', ' ')}:")
        for verb, pmi in verbs[:20]:
            print(f"  {verb}: {pmi:.4f}")


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "part e functions to print results in output"')


# In[ ]:


#main: run in sequence
if __name__ == "__main__":
    path = Path.cwd() / "texts" / "novels"
    print(path)

    #a) Load novels into DataFrame
    #create display dataframe
    #text column is truncated to include first 30 characters and '...' to allow for ease of reading
    df = read_novels(path)
    df_display = df.copy()
    df_display["text"] = df_display["text"].str[:30] + "..."
    df_display["title"] = df_display["title"].str.replace('_', ' ')
    print(df_display[["text", "title", "author", "year"]].to_string(index=False))

    #download data
    nltk.download("cmudict", quiet=True)
    nltk.download("punkt", quiet=True)

    #b)type-token ratios
    print("\nType-Token Ratios:")
    for title, ttr in get_ttrs(df).items():
      print(f" {title.replace('_' ,' ')}: {ttr:.4f}")

    #c)Flesch-Kincaid Grade Level
    print("\nFlesch-Kincaid Grade Level:")
    for title, fk in get_fks(df).items():
      print(f"  {title.replace('_', ' ')}: {fk}")

    #d)load from pickle if parsed, or parse and save
    pickle_path = Path.cwd() / "pickles" / "parsed.pickle"
    if pickle_path.exists():
        print(f"\nLoading parsed DataFrame from {pickle_path}")
        df = pd.read_pickle(pickle_path)
    else:
        print("\nParsing novels with spaCy")
        df = parse(df)

    #create display dataframe
    #truncate text and doc columns for clearer presentation
    print(df.columns.tolist())
    df_display = df.copy()
    df_display["title"] = df_display["title"].str.replace('_', ' ')
    df_display["text"] = df_display["text"].str[:30] + "..."
    df_display["doc"] = df_display["doc"].apply(lambda doc: str([(token.text, token.pos_, token.dep_) for token in list(doc)[:10]]) + "...")
    print(df_display[["text", "title", "author", "year", "doc"]].to_string(index=False))

    #e) subject and verb analysis
    print_top_subjects(df, n=10)
    print_he_verbs(df)
    print_she_verbs(df)


# In[ ]:


import nbformat
from nbconvert import PythonExporter

exporter = PythonExporter()

for part in ['one', 'two', 'three']:
    notebook_path = f'/content/drive/MyDrive/cw-pack-2026/NLP_assignment_part_{part}.ipynb'
    output_path = f'/content/drive/MyDrive/cw-pack-2026/NLP_assignment_part_{part}.py'

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    source, _ = exporter.from_notebook_node(nb)

    with open(output_path, 'w') as f:
        f.write(source)

    print(f"Saved {output_path}")


# In[ ]:


import zipfile, os

PROJECT = '/content/drive/MyDrive/cw-pack-2026'

with zipfile.ZipFile('/content/submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files_list in os.walk(PROJECT):
        dirs[:] = [d for d in dirs if d not in ('pickles', '.git')]
        for file in files_list:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, os.path.dirname(PROJECT))
            z.write(filepath, arcname)

size_mb = os.path.getsize('/content/submission.zip') / (1024 * 1024)
print(f"Zip size: {size_mb:.1f} MB")


# In[ ]:


from google.colab import files
files.download('/content/submission.zip')

