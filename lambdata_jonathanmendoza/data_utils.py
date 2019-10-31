import pandas as pd

def null_report(df):
    """Returns the sum of all nulls from a passed dataframe, as well as percentage of nulls."""

    count = df.isna().sum()
    
    percent_na = count/(count + df.notna().sum())*100

    print(f'There are a total of {count} nulls, which is {percent_na}% of the data.')
    
def train_val_test_split(df, stratifier = None):

    """Splits a passed dataframe into test, validation, and training sets. 
    80/20 split for df going to train to test, then 80/20 from test to validations set. 
    Stratifier should be passed in the form: df['feature']if stratification based 
    on a target is desired """

    train, test = train_test_split(df, train_size=0.8, test_size = 0.2,
                                    stratify = stratifier)
    
    train, val = train_test_split(train, train_size=0.8, test_size = 0.2,
                                    stratify = stratifier)

    return train, test, val
