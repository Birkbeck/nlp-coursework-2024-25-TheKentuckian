import pandas as pd
from pathlib import Path
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

csv_path = Path.cwd() / "p2-texts" / "hansard40000.csv"

def etl_csv():
    major_parties = ['Labour','Conservative','Scottish National Party','Liberal Democrat']

    df = pd.read_csv(csv_path)
    df.loc[df['party'] == 'Labour (Co-op)', 'party'] = 'Labour'
    df.drop(df[~df['party'].isin(major_parties)].index, inplace=True)
    df.drop(df[df['speech_class'] != 'Speech'].index, inplace=True)
    df.drop(df[df['speech'].str.len() < 1000].index, inplace=True)

    return df

def vectorize_speeches(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    return X, y

def vectorize_speeches_including_ngrams(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,3))
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    return X, y

def custom_tokenizer(doc):
        """
        Modelled after the scikit-learn default tokenizer found here:
        https://github.com/scikit-learn/scikit-learn/blob/da08f3d99/sklearn/feature_extraction/text.py#L346
        """

        """
        Whereas the default sklearn tokenizer splits whole words (of at least 2-characters), 
        this approach splits on vowel clusters.
        """
        regex_pattern = r'(?u)\w*?[aeiouAEIOU]+\w*?(?=[aeiouAEIOU]|\b)'
        token_pattern = re.compile(regex_pattern)

        if token_pattern.groups > 1:
            raise ValueError(
                "More than 1 capturing group in token pattern. Only a single "
                "group should be captured."
            )

        return token_pattern.findall(doc)

def vectorize_speeches_custom_tokenizer(df):
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=3000, 
        tokenizer=custom_tokenizer,
        ngram_range=(3,4))
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    return X, y

def get_random_forest_predictions(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators=300, random_state=26)
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)

def get_SVM_predictions(X_train, y_train):
    classifier = SVC(kernel='linear', random_state=26)
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)

def print_prediction_analysis(y_test, predictions):
    score = f1_score(y_test, predictions, average='macro')
    print("F1 score:", score)
    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    
if __name__ == "__main__":
    df = etl_csv()
    print(df.shape)
    
    X, y = vectorize_speeches(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        stratify=y, 
        random_state=26)

    predictions = get_random_forest_predictions(X_train, y_train)
    print("RandomForest Results:")
    print_prediction_analysis(y_test, predictions)
    
    predictions = get_SVM_predictions(X_train, y_train)
    print("SVM Results:")
    print_prediction_analysis(y_test, predictions)

    X, y = vectorize_speeches_including_ngrams(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        stratify=y, 
        random_state=26)

    predictions = get_random_forest_predictions(X_train, y_train)
    print("RandomForest Results (with n-grams):")
    print_prediction_analysis(y_test, predictions)
    
    predictions = get_SVM_predictions(X_train, y_train)
    print("SVM Results (with n-grams):")
    print_prediction_analysis(y_test, predictions)

    X, y = vectorize_speeches_custom_tokenizer(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        stratify=y, 
        random_state=26)

    # achieves .49 F1, higher than default approach
    # predictions = get_random_forest_predictions(X_train, y_train)
    # print("RandomForest Results (with custom tokenizer):")
    # print_prediction_analysis(y_test, predictions)
    
    # achieves .586 F1
    predictions = get_SVM_predictions(X_train, y_train)
    print("SVM Results (with custom tokenizer):")
    print_prediction_analysis(y_test, predictions)

