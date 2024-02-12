import pandas as pd

def df_add_dict(df, dict_new, idx):
    '''Append a dict as a new row in a dataframe.'''
    df_new = pd.DataFrame(dict_new, index=[idx])
    # Concat to current df if passed; otherwise, return new df
    return pd.concat([df, df_new]) if type(df)==pd.core.frame.DataFrame else df_new