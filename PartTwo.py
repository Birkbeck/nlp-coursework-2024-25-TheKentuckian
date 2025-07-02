import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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