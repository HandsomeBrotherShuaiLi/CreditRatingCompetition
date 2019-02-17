import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import lightgbm as lgb
import h2o
import catboost as cbt
import xgboost as xbt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import datetime,numpy as np
def read_file():
    trainset=pd.read_csv('data/train_dataset/train_dataset.csv')
    testset=pd.read_csv('data/test_dataset/test_dataset.csv')
    cols=['var_'+str(i) for i in range(30)]
    trainset_temp=trainset.copy()
    testset_temp=testset.copy()
    trainset_temp.columns=cols
    testset_temp.columns=cols[:-1]
    trainset_temp.to_csv('data/train_dataset/train_dataset_temp.csv',index=False)
    testset_temp.to_csv('data/test_dataset/test_dataset_temp.csv',index=False)


    print(trainset.columns,len(trainset))
    print(testset.columns,len(testset))
    #都是50000个
    train_nan=[]
    test_nan=[]
    print(trainset.isnull().any())
    for i in trainset.columns:
        print(i+' type is {}'.format(trainset[i].dtype))

    for i in testset.columns:
        print(i+' type is {}'.format(testset[i].dtype))
        #都是int64 float64类型 nice!
        # 检查是不是有缺失值
        print(i+': the number of nan value is {}'.format(testset[i].isnull().sum()))
        if testset[i].isnull().sum()!=0:
            test_nan.append(i)
    print(train_nan,test_nan)
    #无缺失值，nice!

    #画图
    plt.title('ranking score')
    plt.xlabel('score value')
    plt.ylabel('people num')
    plt.bar(trainset['信用分'].value_counts().index.values,trainset['信用分'].value_counts().values)
    plt.savefig('data/信用分.png')

def training(name):
    """
    使用四种训练方式，h2o,xgboost,catboost,lightboost
    :return:
    """
    trainset = pd.read_csv('data/train_dataset/train_dataset.csv')
    testset = pd.read_csv('data/test_dataset/test_dataset.csv')
    features=[c for c in trainset.columns if c not in ['用户编码','信用分']]
    target=trainset['信用分']
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2019)
    oof=np.zeros(len(trainset))
    predictions=np.zeros(len(testset))
    features_importance_df=pd.DataFrame()
    # lgbparam={
    #     'num_leaves': 15,
    #     'max_bin': 28,
    #     'min_data_in_leaf': 10,
    #     'learning_rate': 0.001,
    #     'min_sum_hessian_in_leaf': 1e-3,
    #     'bagging_fraction': 0.9,
    #     'bagging_freq': 2,
    #     'feature_fraction': 0.9,
    #     'lambda_l1': 4.972,
    #     'lambda_l2': 2.277,
    #     'min_gain_to_split': 0.65,
    #     'max_depth': 11,
    #     'save_binary': True,
    #     'seed': 1337,
    #     'feature_fraction_seed': 1337,
    #     'bagging_seed': 1337,
    #     'drop_seed': 1337,
    #     'data_random_seed': 1337,
    #     'objective': 'regression',
    #     'boosting_type': 'gbdt',
    #     'verbose': 1,
    #     'metric': 'auc',
    #     'is_unbalance': False,
    #     'boost_from_average': False,
    #     'is_provide_training_metric':True
    # }
    lgbparam={
        'num_leaves': 10,
        'max_bin': 10,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': '',
        'is_unbalance': False,
        'boost_from_average': False,
    }
    print('start KFold {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for fold_,(trn_index,val_index) in enumerate(skf.split(trainset.values,target.values)):
        print('fold {}'.format(fold_+1))
        print('start time {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        trn_data=lgb.Dataset(trainset.iloc[trn_index][features],label=target.iloc[trn_index])
        print('train data number is {}'.format(len(trn_index)))
        val_data=lgb.Dataset(trainset.iloc[val_index][features],label=target.iloc[val_index])

        clf=lgb.train(params=lgbparam,train_set=trn_data,num_boost_round=5000,valid_sets=[trn_data,val_data],
                      verbose_eval=1000,early_stopping_rounds=100)
        oof[val_index]=clf.predict(trainset.iloc[val_index][features],num_iteration=clf.best_iteration)
        fold_importance_df=pd.DataFrame()
        fold_importance_df['features']=features
        fold_importance_df['importance']=clf.feature_importance()
        fold_importance_df['fold']=fold_+1
        features_importance_df=pd.concat([features_importance_df,fold_importance_df],axis=0)
        predictions+=clf.predict(testset[features],num_iteration=clf.best_iteration)/5
    print('cross validation score : {:<8.5f}'.format(roc_auc_score(target,oof)))
    cols=features_importance_df[['features','importance']].groupby('features').mean().sort_values(by='importance',ascending=False).index
    best_features=features_importance_df.loc[features_importance_df.feature.isin(cols)]
    plt.figure(figsize=(14,26))
    import seaborn as sns
    sns.barplot(x='importance',y='feature',data=best_features.sort_values(by="importance",ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('data/lgbm_features_scores.png')
    sub_df=pd.DataFrame({'id':testset['用户编码'].values})
    sub_df['score']=predictions
    sub_df.to_csv('data/lgbm_submission_{}.csv'.format(name),index=False)

def train_with_h2o_full_data(time):
    import h2o
    from h2o.automl import H2OAutoML
    h2o.init()
    h2o.remove_all()
    raw_train_df=h2o.import_file(path='data/train_dataset/train_dataset_temp.csv')
    raw_test_df=h2o.import_file(path='data/test_dataset/test_dataset_temp.csv')
    train_df=raw_train_df[:,1:]
    y='var_29'
    col_list=train_df.columns
    print(col_list)
    x=col_list[:-1]
    print('x:',x)
    print('y:',y)
    splits=train_df.split_frame(ratios = [0.9], seed = 1)
    train=splits[0]
    test=splits[1]
    #part 50000行
    # aml1 = H2OAutoML(max_runtime_secs=time,balance_classes=False,stopping_tolerance=0.005,stopping_rounds=50,sort_metric='MAE',stopping_metric='MAE',seed=2019, project_name="part_data_train")
    # aml1.train(x=x,y=y,training_frame=train,leaderboard_frame=test)
    #full data train
    aml2 = H2OAutoML(max_runtime_secs=time,balance_classes=False,stopping_tolerance=0.005,stopping_rounds=50,sort_metric='MAE',stopping_metric='MAE', seed=2019, project_name="full_data_train")
    aml2.train(x=x, y=y, training_frame=train_df)
    # path1=h2o.save_model(model=aml1,path='models',force=True)
    # path2=h2o.save_model(model=aml2,path='models',force=True)
    # print(aml1.leaderboard)
    print('++++++++++++++++++++++')
    print(aml2.leaderboard)
    # ans1=aml1.predict(raw_test_df[:,1:])
    ans2=aml2.predict(raw_test_df[:,1:])
    # print(ans1)
    print(ans2)
    # ans1=ans1.as_data_frame()
    ans2=ans2.as_data_frame()
    # ans1.to_csv('data/ans1.csv',index=False)
    ans2.to_csv('data/ans2_time_{}.csv'.format(str(time)),index=False)

    # res1=pd.DataFrame()
    temp=pd.read_csv('data/test_dataset/test_dataset_temp.csv')
    # res1['id']=temp['var_0']
    # res1['score']=ans1.values
    # res1.to_csv('data/h2o_pred_submission_v1.csv',index=False)

    res2=pd.DataFrame()
    res2['id']=temp['var_0']
    res2['score']=ans2.values
    res2['score'] = res2['score'].apply(lambda x: int(x) if (x - int(x)) < 0.5 else int(x) + 1)
    res2.to_csv('data/h2o_pred_submission_int_v2_time_{}.csv'.format(time),index=False)

def train_with_h2o_part_data(time):
    import h2o
    from h2o.automl import H2OAutoML
    h2o.init()
    h2o.remove_all()
    raw_train_df=h2o.import_file(path='data/train_dataset/train_dataset_temp.csv')
    raw_test_df=h2o.import_file(path='data/test_dataset/test_dataset_temp.csv')
    train_df=raw_train_df[:,1:]
    y='var_29'
    col_list=train_df.columns
    print(col_list)
    x=col_list[:-1]
    print('x:',x)
    print('y:',y)
    splits=train_df.split_frame(ratios = [0.8], seed = 1)
    train=splits[0]
    test=splits[1]
    #part 50000行
    aml1 = H2OAutoML(max_runtime_secs=time,balance_classes=False,stopping_tolerance=0.005,stopping_rounds=50,sort_metric='MAE',stopping_metric='MAE',seed=2019, project_name="part_data_train")
    aml1.train(x=x,y=y,training_frame=train,leaderboard_frame=test)
    #full data train
    # aml2 = H2OAutoML(max_runtime_secs=time,balance_classes=False,stopping_tolerance=0.005,stopping_rounds=50,sort_metric='MAE',stopping_metric='MAE', seed=2019, project_name="full_data_train")
    # aml2.train(x=x, y=y, training_frame=train_df)
    # path1=h2o.save_model(model=aml1,path='models',force=True)
    # path2=h2o.save_model(model=aml2,path='models',force=True)
    print(aml1.leaderboard)
    # print('++++++++++++++++++++++')
    # print(aml2.leaderboard)
    ans1=aml1.predict(raw_test_df[:,1:])
    # ans2=aml2.predict(raw_test_df[:,1:])
    print(ans1)
    # print(ans2)
    ans1=ans1.as_data_frame()
    # ans2=ans2.as_data_frame()
    ans1.to_csv('data/ans1_time_{}.csv'.format(str(time)),index=False)
    # ans2.to_csv('data/ans2_time_{}.csv'.format(str(time)),index=False)

    res1=pd.DataFrame()
    temp=pd.read_csv('data/test_dataset/test_dataset_temp.csv')
    res1['id']=temp['var_0']
    res1['score']=ans1.values
    res1['score']=res1['score'].apply(lambda x: int(x) if (x - int(x)) < 0.5 else int(x) + 1)
    res1.to_csv('data/h2o_pred_submission_v1_time_{}.csv'.format(time),index=False)

    # res2=pd.DataFrame()
    # res2['id']=temp['var_0']
    # res2['score']=ans2.values
    # res2['score'] = res2['score'].apply(lambda x: int(x) if (x - int(x)) < 0.5 else int(x) + 1)
    # res2.to_csv('data/h2o_pred_submission_int_v2_time_{}.csv'.format(time),index=False)

def makeup(time):
    # res1=pd.read_csv('data/h2o_pred_submission_v1.csv')
    res2=pd.read_csv('data/h2o_pred_submission_v2_time_{}.csv'.format(time))
    # res1['score']=res1['score'].apply(lambda x:int(x) if (x-int(x))<0.5 else int(x)+1)
    res2['score']=res2['score'].apply(lambda x:int(x) if (x-int(x))<0.5 else int(x)+1)
    # res1.to_csv('data/h2oSubmission_final_v1.csv',index=False)
    res2.to_csv('data/h2oSubmission_final_v2_time_{}.csv'.format(time),index=False)

if __name__=='__main__':
    train_with_h2o_full_data(72000)
