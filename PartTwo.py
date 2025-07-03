import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

csv_path = Path.cwd() / "p2-texts" / "hansard40000.csv"

def etl_csv():
    df = pd.read_csv(csv_path)
    
    df.loc[df['party'] == 'Labour (Co-op)', 'party'] = 'Labour'
    
    major_parties = ['Labour','Conservative','Scottish National Party','Liberal Democrat']
    df.drop(df[~df['party'].isin(major_parties)].index, inplace=True)

    df.drop(df[df['speech_class'] != 'Speech'].index, inplace=True)
    print(df['speech_class'].unique())

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

