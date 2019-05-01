from sklearn.model_selection import train_test_split
import pandas as pd
import sys, os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pickle
def make_train_test(data_path):
    seed=6237
    load_df=pd.read_csv(data_path)

    if 'HELOC' in data_path:
        label_name='RiskPerformance'
    elif 'UCI_Credit' in data_path:
        label_name='default.payment.next.month'
    elif 'statlog' in data_path:
        label_name = 'Credit'
    else:
        print('label name not vaild!')
        sys.exit(0)

    df_x=load_df.drop([label_name],axis=1)
    df_x=df_x.drop([0,1,2],axis=0)

    df_y=load_df[label_name]
    df_y=df_y.drop([0,1,2],axis=0)

    tr_data, te_data, tr_label, te_label=train_test_split(df_x, df_y, random_state=seed)

    return tr_data, te_data,tr_label, te_label


def fico_data_preprocessing(data_path):
    file_name_v1='../Dataset/HELOC_ExternalRemoved.csv'
    file_name_v2='../Dataset/HELOC_-9Removed.csv'
    file_name_v3='../Dataset/HELOC_allRemoved.csv'
    if os.path.exists(file_name_v3):
        return file_name_v3

    load_df=pd.read_csv(data_path)

    label_name='ExternalRiskEstimate'

    df=load_df.drop([label_name],axis=1)

    del_list=list()
    for row in load_df.iterrows():
        index,value =row
        count=value.value_counts()
        if count[0]==23:
            count+=1
            del_list.append(index)
            load_df=load_df.drop([index],axis=0)

    df2 = load_df
    df3=df2.drop([label_name],axis=1)


    df.to_csv(file_name_v1,index=False)
    df2.to_csv(file_name_v2,index=False)
    df3.to_csv(file_name_v3,index=False)

    return file_name_v3

def UCI_data_preprocessing(data_path):

    file_name='../Dataset/UCI_Credit_Card_IDRemoved.csv'
    if os.path.exists(file_name):
        return file_name

    load_df=pd.read_csv(data_path)
    label_name='ID'

    df=load_df.drop([label_name],axis=1)

    df.to_csv(file_name,index=False)

    return file_name


def make_explain_data(te_data, prob, model_name, data_path):
    load_df=te_data
    load_df_copy=te_data
    df=load_df.values


    #########################
    # genearating valueSim data

    #Normalizing each column
    scaler=MinMaxScaler()
    scaled_value=scaler.fit_transform(df)
    load_df=pd.DataFrame(scaled_value)
    load_df=load_df.values

    #base index for valueSim
    base_index=random.randrange(1, load_df.shape[0])
    base=np.reshape(load_df[base_index],(1,-1))
    load_df=np.delete(load_df,base_index,0)

    dist_value=np.empty((load_df.shape[0]))
    i=0
    for row in load_df:
        row=np.reshape(row,(1,-1))

        dist_value[i]=euclidean_distances(row,base)
        i += 1

    min_sort=np.argsort(dist_value)
    base_value=load_df_copy.iloc[base_index]

    df_valueSim=pd.DataFrame([base_value.values.T],columns=np.asarray(base_value.index))
    for i in range(10):
        tmp_value = load_df_copy.iloc[min_sort[i]]

        tmp=pd.DataFrame([tmp_value.values.T],columns=np.asarray(base_value.index))
        df_valueSim=pd.concat([df_valueSim,tmp],ignore_index=True)


    #########################
    #genearating probsim data
    prob=prob[:,0]
    prob_index=np.argsort(prob)


    base_index=random.randrange(prob.shape[0]/2,prob.shape[0])
    while( prob[prob_index[base_index]] <= 0.6):
        base_index = random.randrange(prob.shape[0] / 2, prob.shape[0])

    base_value=load_df_copy.iloc[prob_index[base_index]]

    df_probSim=pd.DataFrame([base_value.values.T],columns=np.asarray(base_value.index))

    for i in range(10):
        tmp_value = load_df_copy.iloc[prob_index[base_index+i+1]]

        tmp = pd.DataFrame([tmp_value.values.T], columns=np.asarray(base_value.index))
        df_probSim = pd.concat([df_probSim, tmp], ignore_index=True)






    #########################
    #genearating random data
    base_index_s=random.randrange(1,load_df.shape[0])
    base_value=load_df_copy.iloc[prob_index[base_index_s]]

    df_random=pd.DataFrame([base_value.values.T],columns=np.asarray(base_value.index))

    for i in range(10):
        base_index_s = random.randrange(1, load_df.shape[0])
        tmp_value = load_df_copy.iloc[base_index_s]

        tmp = pd.DataFrame([tmp_value.values.T], columns=np.asarray(base_value.index))
        df_random = pd.concat([df_random, tmp], ignore_index=True)


    if 'heloc' in data_path:
        if 'Random_Forest'==model_name:
            df_valueSim.to_csv('../Explain_te_data/heloc/RF_explain_valueSim.csv', index=False)
            df_probSim.to_csv('../Explain_te_data/heloc/RF_explain_probSim.csv', index=False)
            df_random.to_csv('../Explain_te_data/heloc/RF_explain_random.csv', index=False)
        elif 'XGboost' ==model_name:
            df_valueSim.to_csv('../Explain_te_data/heloc/XGboost_explain_valueSim.csv', index=False)
            df_probSim.to_csv('../Explain_te_data/heloc/XGboost_explain_probSim.csv', index=False)
            df_random.to_csv('../Explain_te_data/heloc/XGboost_explain_random.csv', index=False)

    elif 'UCI_Credit_Card' in data_path:
        if 'Random_Forest' == model_name:
            df_valueSim.to_csv('../Explain_te_data/UCI_Credit_Card/RF_explain_valueSim.csv', index=False)
            df_probSim.to_csv('../Explain_te_data/UCI_Credit_Card/RF_explain_probSim.csv', index=False)
            df_random.to_csv('../Explain_te_data/UCI_Credit_Card/RF_explain_random.csv', index=False)
        elif 'XGboost' ==model_name:
            df_valueSim.to_csv('../Explain_te_data/UCI_Credit_Card/XGboost_explain_valueSim.csv', index=False)
            df_probSim.to_csv('../Explain_te_data/UCI_Credit_Card/XGboost_explain_probSim.csv', index=False)
            df_random.to_csv('../Explain_te_data/UCI_Credit_Card/XGboost_explain_random.csv', index=False)

    elif 'statlog' in data_path:
        if 'Random_Forest' == model_name:
            df_valueSim.to_csv('../Explain_te_data/statlog/RF_explain_valueSim.csv', index=False)
            df_probSim.to_csv('../Explain_te_data/statlog/RF_explain_probSim.csv', index=False)
            df_random.to_csv('../Explain_te_data/statlog/RF_explain_random.csv', index=False)
        elif 'XGboost' ==model_name:
            df_valueSim.to_csv('../Explain_te_data/statlog/XGboost_explain_valueSim.csv', index=False)
            df_probSim.to_csv('../Explain_te_data/statlog/XGboost_explain_probSim.csv', index=False)
            df_random.to_csv('../Explain_te_data/statlog/XGboost_explain_random.csv', index=False)


        """
        #######
        # verify all samples having similar probability
        if 'heloc' in data_path:
            data_n='heloc'
            if 'Random_Forest' == model_name:
                model_n='Random_Forest'
                model_n2='RF'
            elif 'XGboost' == model_name:
                model_n="XGboost"
                model_n2 = "XGboost"
        elif 'UCI_Credit_Card' in data_path:
            data_n = 'UCI_Credit_Card'
            if 'Random_Forest' == model_name:
                model_n='Random_Forest'
                model_n2='RF'

            elif 'XGboost' == model_name:
                model_n="XGboost"
                model_n2 = "XGboost"

        model = pickle.load(open('trained_model/'+model_n+'.sav', 'rb'))

        te_data = pd.read_csv('../Test_data/'+data_n+'/'+model_n2+'_explain_probSim.csv')

        prob2 = model.predict_proba(te_data)
        print prob2

        print prob[prob_index[base_index]]
        print prob[prob_index[base_index + 1]]
        print prob[prob_index[base_index + 2]]
        print prob[prob_index[base_index + 3]]
        print prob[prob_index[base_index + 4]]
        print prob[prob_index[base_index + 5]]
        print prob[prob_index[base_index + 6]]
        print prob[prob_index[base_index + 7]]
        print prob[prob_index[base_index + 8]]

        ##############
        """

def get_data(data_path):

    # Read file in csv format
    train = pd.read_csv(data_path+'_train.csv')
    test = pd.read_csv(data_path+'_test.csv')

    feature, target = get_feature(data_path, train)

    #train_x, test_x, train_y, test_y
    return train[feature], test[feature], train[target], test[target]

def get_feature(data_path, x):

    if 'HELOC' in data_path:
        label_name = 'RiskPerformance'
    elif 'UCI_Credit' in data_path:
        label_name = 'default.payment.next.month'
    else:
        print('label name not vaild!')
        sys.exit(0)

    features = [c for c in x.columns if c != label_name]

    return features, label_name