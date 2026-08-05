#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Part 3

import os
import random

import nltk
import spacy
from pathlib import Path
import numpy as np
import requests
import time

import math
import re
from collections import Counter
import pandas as pd
from spacy.tokens import Doc

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score

random.seed(26)
np.random.seed(26)

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


# In[26]:


get_ipython().system('pip install openai python-dotenv')
import os
os.environ["OPENROUTER_API_KEY"] = input("Enter API key: ")


# In[27]:


#Part a) Model
#Model = Llama-3.1-8B-Instruct
#Access = OpenRouter

#Generation parameters:
  #temperature = 0
    #Temperature was set to 0 for a fully deterministic output, as no randomness is required for retrieving the single highest-probability label. This ensures consistent predictions across repeated runs.
  #max_tokens = 10
    #Max_tokens was limited to 10, because the expected output is only one party label. Restricting the token budget reduces the chance of the model attempting to produce additional, but unnecessary, explanation text.

#Model choice rationale:
  #Llama-3.1-8B-Instruct was selected because it provides a good balance between efficiency and performance. With 8B parameters, this model is small enough to run quickly across many test instances, while still providing the capture of political distinctions from parliamentary speech. It is instruction-tuned, which makes it more reliable for following prompt constraints like returning exactly one lavel. Larger models may provide more accuracy, but it would increase latency and API cost with no guarantee of improvement in this classification task.


# In[28]:


#configuration
#key from openrouter
from openai import OpenAI
import os

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model = "meta-llama/llama-3.1-8b-instruct",
    messages = [{"role": "user", "content": "Explain gravity simply"}]
)

print(response.choices[0].message.content)

MODEL = "meta-llama/llama-3.1-8b-instruct"
#deterministic
TEMPERATURE = 0
#one label word needed
MAXIMUM_TOKENS = 10
#truncate before sending to api
MAXIMUM_SPEECH_CHARACTERS = 2000
#labelled examples per party in the few-shot prompt
N_FEW_SHOT = 2


# In[29]:


#read and clean dataset same way as part 2
df = pd.read_csv(os.path.join(PROJECT, "texts", "hansard500.csv"))

#replace Labour (Co-op) -> Labour
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

#keep only four most common parties and remove Speaker
df = df[df['party'] != 'Speaker']
top_four = df['party'].value_counts().head(4).index
df = df[df['party'].isin(top_four)]

#keep rows with speech_class as 'Speech'
df = df[df['speech_class'] == 'Speech']

#keep rows where speech is at least 1000 characters
df = df[df['speech'].str.len() >= 1000]

labels = sorted(df["party"].unique())

print("Dataset shape:", df.shape)
print("Labels:", labels)


# In[30]:


#split data same as part 2
X = df["speech"]
y = df["party"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 26)

print("Train:", len(X_train), "Test:", len(X_test))


# In[31]:


#call API
def call_llm(prompt: str) -> str:
  response = requests.post(
      "https://openrouter.ai/api/v1/chat/completions",
      headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"},
      json = {"model": MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": TEMPERATURE,
              "max_tokens": MAXIMUM_TOKENS
              },
      timeout = 30)

  if response.status_code != 200:
      print("STATUS:", response.status_code)
      print("RESPONSE:", response.text)
      return "Unknown"

  return response.json()["choices"][0]["message"]["content"].strip()


# In[32]:


#parse the labels
def parse_label(raw: str, labels: list) -> str:
  #extract party label from the models raw output and strips punctuation and makes it case-insensitive
  #fallback -> unknown if it cannot find label
  cleaned = raw.lower().strip().replace(":", "").replace(".", "").replace("\n", " ")

  for label in labels:
    if label.lower() in cleaned:
      return label

  return "Unknown"


# In[33]:


#b)
#zero-shot classification
print("Part b) Zero-shot classification")

#template
zero_shot_prompt = """
You are a political classifier.

Your task is to identify which UK political party the speaker belongs to.

Choose ONLY from:
{labels}

Rules:
- Output only one of the following labels: {labels}.
- Do not explain.
- Do not add punctuation.

Speech:
\"\"\"{speech}\"\"\"

Answer:
"""

#classification
def classify_zero_shot(speech: str) -> str:
    prompt = zero_shot_prompt.format(
        labels = ", ".join(labels),
        speech = speech[:MAXIMUM_SPEECH_CHARACTERS])

    raw = call_llm(prompt)
    return parse_label(raw, labels)

#run evaluation
zero_shot_predictions = []

for i, speech in enumerate(X_test):
  prediction = classify_zero_shot(speech)
  zero_shot_predictions.append(prediction)

  if (i + 1) % 10 == 0:
    print("Zero-shot:", i + 1, "/", len(X_test), "\n")

  time.sleep(0.3) #delays to respect rate limits

zero_shot_macro_f1 = f1_score(y_test, zero_shot_predictions, average = "macro", zero_division = 0)

print("Zero-shot prompt template:\n")
print(zero_shot_prompt.format(labels = ", ".join(labels), speech = "<speech text is truncated to 2000 characters>"))
print("\nZero-shot F1:", zero_shot_macro_f1)
print(classification_report(y_test, zero_shot_predictions, labels=labels, zero_division = 0))


# In[34]:


#c)
#Few-shot classification
train_df = pd.DataFrame({"speech": X_train.values, "party": y_train.values})

#create length column (number of characters)
train_df = train_df.copy()
train_df["length"] = train_df["speech"].str.len()

def select_few_shot_examples(train_df, n = 2):
  examples = {}

  for party in labels:
    party_df = train_df[train_df["party"] == party].copy()

    #select longer speeches
    selected = party_df.sort_values("length", ascending = False).head(n)["speech"].tolist()

    examples[party] = selected

  return examples

few_shot_examples = select_few_shot_examples(train_df, N_FEW_SHOT)


# In[35]:


#build example block for few-shot
def build_examples_block(examples, max_chars = 300):
  lines = []

  for party, speeches in examples.items():
    for s in speeches:
      lines.append(f"{party}: {s[:max_chars]}")

  return "\n".join(lines)

examples_block = build_examples_block(few_shot_examples)


# In[36]:


#c)
#few-shot classification prompt
#template
print("Part c) Few-shot classification")

few_shot_prompt = """
You are a political classifier.

Your task is to identify which UK political party the speaker belongs to.

Choose ONLY from:
{labels}

Here are examples:

{examples}

Rules:
- Output only the party name.
- Do not explain.
- Do not add punctuation.

Speech:
\"\"\"{speech}\"\"\"

Answer:
"""

#classify
def classify_few_shot(speech: str) -> str:
  prompt = few_shot_prompt.format(labels = ", ".join(labels), examples = examples_block, speech = speech[:MAXIMUM_SPEECH_CHARACTERS])

  raw = call_llm(prompt)
  return parse_label(raw, labels)

#run evaluation
few_shot_predictions = []

for i, speech in enumerate(X_test):
  prediction = classify_few_shot(speech)
  few_shot_predictions.append(prediction)

  if (i + 1) % 10 == 0:
    print("Few-shot:", i + 1, "/", len(X_test))

  time.sleep(0.3)

few_shot_macro_f1 = f1_score(y_test, few_shot_predictions, average = "macro", zero_division = 0)

print("Few-shot prompt template:\n")
print(few_shot_prompt.format(labels=", ".join(labels), examples=examples_block, speech="<speech text is truncated to 2000 characters>"))
print(f"Example selection strategy: {N_FEW_SHOT} longest speeches per party were selected from the training set. Any longer speeches are assumed to be more likely to contain more ideological content, policy references, and party-specific language. Each example has been truncated to 300 characters when inserted into the prompt.")
print("\nFew-shot F1:", few_shot_macro_f1)
print(classification_report(y_test, few_shot_predictions, labels=labels, zero_division = 0))


# In[43]:


#compare
print("\nPart d) Comparison")
print("Zero-shot", zero_shot_macro_f1)
print("Few-shot:", few_shot_macro_f1)

delta = few_shot_macro_f1 - zero_shot_macro_f1
print("Delta", delta)
print("\nExplanation: \nIn this task, the few-shot prompt performed slightly better than the zero-shot prompt, achieving a macro-average F1 score of 0.8286 compared to 0.7975 for zero-shot, giving a difference of +0.0312. This suggests that adding labelled examples provided a slight improvement in classification performance. \nHowever, the improvement is modestly small, indicating that few-shot prompting did not significantly change model behaviour. A likely reason for this is the example selection strategy. For few-shot prompting, the two longest speeches from each party in the training set were selected - under the assumption that longer speeches would contain more ideological language, policy references, and stronger party-specific language. However, these examples may have introduced noise or excessively specific signals that did not generalise well to unseen test speeches. \nIt should also be noted that the Democratic Unionist Party had zero instances in the test set (support = 0), meaning macro-F1 is computed across four classes, but only three actually contribute meaningful scores. This can reduce the stability of macro-F1 as a metric here, as one class contributes a score of 0.00 despite having no real test samples. As a result, weighted F1 could provide a more reliable indicator of the performance because it accounts for class distribution. \nThe zero-shot prompt was designed with strong constraints, explicitly restricting outputs to the predefined label set and enforcing a single-label response without explanations or extra formatting. The few-shot prompt used the same constraints, but additionally included the labelled examples selected to guide classification behaviour. \nOverall, the results suggest that few-shot prompting provided only a slight performance improvement over zero-shot prompting in this task, but the gain is limited. This indicates that prompt quality and example selection strategy are more important than simply adding more examples. More effective few-shot performance could perhaps be achieved by using shorter, more representative examples that better match the style or distribution of the test data speeches.")


# In[38]:


#predictions
results_df = pd.DataFrame({"Actual": y_test.values, "Zero_shot_prediction": zero_shot_predictions, "Few_shot_prediction": few_shot_predictions})
print(results_df)

