import pandas as pd
from pathlib import Path

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

if __name__ == "__main__":
    df = etl_csv()
    print(df.shape)