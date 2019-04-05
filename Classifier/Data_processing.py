from sklearn.model_selection import train_test_split
import pandas as pd

def make_train_test(data_path):
    seed=6237
    load_df=pd.read_csv(data_path)

    label_name='RiskPerformance'
    df_x=load_df.drop([label_name],axis=1)
    df_y=load_df[label_name]
    tr_data, te_data, tr_label, te_label=train_test_split(df_x, df_y, random_state=seed)

    return tr_data, te_data, tr_label, te_label
