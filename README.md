# Natural Language Processing Project

Project applying text processing, machine learning, and large language model techniques to analyse and classify literary texts and political speeches.

## Project Components

### Part One: Syntax and Style Analysis

Analyses a collection of 19th-century novels using NLP techniques, including:

- Text processing with pandas and NLTK
- Type-token ratio and readability analysis
- Flesch-Kincaid readability scoring
- spaCy parsing and syntactic analysis
- Pointwise Mutual Information (PMI) analysis of verb-subject relationships

### Part Two: Feature Extraction and Classification

Develops machine learning classifiers to predict political party from parliamentary speeches using the Hansard dataset. Methods include:

- TF-IDF feature extraction
- Text preprocessing and custom tokenisation
- Random Forest and Support Vector Machine classifiers
- Model evaluation using F1-score and classification reports

### Part Three: Zero-shot and Few-shot LLM Classification

Investigates the use of large language models for political speech classification through:

- Zero-shot prompting
- Few-shot prompting with labelled examples
- Evaluation and comparison of LLM classification performance
