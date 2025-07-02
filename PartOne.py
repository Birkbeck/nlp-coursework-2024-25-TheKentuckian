#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution 
# is a suggested but not mandatory approach. You can use a different approach if you like, 
# as long as you clearly answer the questions and communicate your answers clearly.

import pandas as pd
import nltk
import spacy
import spacy.cli
from pathlib import Path
from collections import Counter
import math

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000

nltk.download('cmudict')
nltk.download('punkt_tab')

def fk_level(parsed_text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    
    sentences = list(parsed_text.sents)
    sentences_count = len(sentences)
    
    words = [token for token in parsed_text if token.is_alpha and not token.is_space]
    words_count = len(words)
    
    syllables_count = sum(count_syl(token.text.lower(), d) for token in words)
    
    if sentences_count == 0 or words_count == 0:
        return 0.0
    
    fk_score = 0.39 * (words_count / sentences_count) + 11.8 * (syllables_count / words_count) - 15.59
    return fk_score

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    
    word = word.lower()
    
    if word in d:
        return len(d[word][0])
    
    vowels = 'aeiouy'
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    return max(1, syllable_count)

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    df = pd.DataFrame(columns=['text','title','author','year'])
    df
    # sort the dataframe by the year column before returning it, resetting or
    # ignoring the dataframe index.
    print(path)
    for filename in path.iterdir():
        # read text from file
        try:
            with open(filename, newline='', encoding='utf-8') as openbook:
                text = openbook.readlines()
                nameparts = filename.name.replace(".txt","").split('-')
                # sample filename: The_Secret_Garden-Burnett-1911
                title = nameparts[0].replace('_','')
                author = nameparts[1]
                year = nameparts[2]
                df.loc[len(df)] = [text, title, author, year]
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
    
    return df.sort_values('year')
    
def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    parsed_docs = []
    
    for i, row in df.iterrows():
        print("Starting the parse of ",row['title'])
        doc = nlp(''.join(row['text']))
        parsed_docs.append(doc)
    
    df['parsed'] = parsed_docs
    
    store_path.mkdir(exist_ok=True)
    output_path = store_path / out_name
    df.to_pickle(output_path)
    
    return df

def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    # Tokenize the text using the NLTK library only.
    # Do not include punctuation as tokens, and ignore case when counting types.
    tokenized = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokenized if token.isalpha()]
    types = set(tokens)
    type_count = len(types)
    token_count = len(tokens)

    return type_count / token_count

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        text = ''.join(row['text'])
        results[row["title"]] = nltk_ttr(text)
    return results

def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        parsed_text = row["parsed"]
        results[row["title"]] = round(fk_level(parsed_text, cmudict), 4)
    return results

def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    
    verb_subject_pairs = []
    subjects = []
    verbs = []
    
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.lemma_.lower()
            verbs.append(verb)
            
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child.text.lower()
                    subjects.append(subject)
                    verb_subject_pairs.append((verb, subject))
    
    if len(verb_subject_pairs) == 0:
        return []
    
    pairs_count = len(verb_subject_pairs)
    verb_counts = Counter(verbs)
    subject_counts = Counter(subjects)
    pair_counts = Counter(verb_subject_pairs)
    
    pmi_scores = []
    target_verb_lower = target_verb.lower()
    
    for (verb, subject), this_pair_count in pair_counts.items():
        if verb == target_verb_lower:
            verb_subject_percentage = this_pair_count / pairs_count
            verb_percentage = verb_counts[verb] / len(verbs)
            subject_percentage = subject_counts[subject] / len(subjects)
            
            if verb_percentage > 0 and subject_percentage > 0:
                pmi = math.log2(verb_subject_percentage / (verb_percentage * subject_percentage))
                pmi_scores.append((subject, pmi))
    
    return sorted(pmi_scores, key=lambda x: x[1], reverse=True)[:5]

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    
    subjects = []
    
    for token in doc:
        if token.lemma_.lower() == verb.lower() and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subjects.append(child.text.lower())
    
    subject_counts = Counter(subjects)
    return subject_counts.most_common(5)

def adjective_counts(df):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""

    results = {}
    for i, row in df.iterrows():
        adjectives = []
        
        for token in row["parsed"]:
            if token.pos_ == "ADJ" and not token.is_space:
                adjectives.append(token.lemma_.lower())
        
        adjective_counter = Counter(adjectives)
        results[row["title"]] = adjective_counter.most_common(5)
    return results

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """

    """ Uncomment this block to regenerate the pickle file
    
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    
    df = parse(df)
    """

    df = pd.read_pickle(Path.cwd() / "pickles" / "parsed.pickle")
    print(df.head())
    print(get_ttrs(df))
    
    print(get_fks(df))
    print(adjective_counts(df))
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

