#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution 
# is a suggested but not mandatory approach. You can use a different approach if you like, 
# as long as you clearly answer the questions and communicate your answers clearly.

import pandas as pd
import nltk
import spacy
import spacy.cli
from pathlib import Path

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass


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
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    # Tokenize the text using the NLTK library only.
    # Do not include punctuation as tokens, and ignore case when counting types.
    num_tokens = nltk.tokenizer(text)
    # text_count = ?


    return text_count / num_tokens

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results

def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

