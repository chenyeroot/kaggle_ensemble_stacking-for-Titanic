# x_train = x_train.drop(["Ticket_Number"],axis=1)
# 用xgboost描述每个模型输出的特征重要度
def modelMetrics(clf,x_train,y_train,isCv=True,cv_folds=5,early_stopping_rounds=300):  
    if isCv:  
        xgb_param = clf.get_xgb_params()
        # 构建xgb矩阵  
        xgtrain = xgb.DMatrix(x_train,label=y_train)  
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=clf.get_params()['n_estimators'],nfold=cv_folds,metrics='auc',
                          early_stopping_rounds=early_stopping_rounds) 
        clf.set_params(n_estimators=cvresult.shape[0])  
  
    # 训练
    clf.fit(x_train,y_train,eval_metric='auc')  
  
    # 预测  
    train_predictions = clf.predict(x_train)  
    train_predprob = clf.predict_proba(x_train)[:,1]  # 1的概率 
    # 打印  
    print("==================模型报告==================")  
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train,train_predictions))  
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train,train_predprob))  
  
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)  
    feat_imp.plot(kind='bar',title='Feature importance')  
    plt.ylabel('Feature Importance Score') 
    plt.show()

# 初始化参数 
xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,
                     colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=2017)    
# 确定学习速率和tree_based 参数调优的估计器数目
modelMetrics(xgb1,x_train,y)

# 用几个模型筛选出较为重要的特征
def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
        # 随机森林
        rf_est = RandomForestClassifier(random_state=42)
        rf_param_grid = {'n_estimators': [100,300], 'min_samples_split': [2, 3], 'max_depth': [15,20]}
        rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=-1, cv=10, verbose=1)
        rf_grid.fit(titanic_train_data_X,titanic_train_data_Y)
        #将feature按Importance排序
        feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X), 
                                              'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
        features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
        print('---------------------------------------')
        print('Sample 25 Features from RF Classifier:')
        print(str(features_top_n_rf[:25]))

        # AdaBoost
        ada_est = AdaBoostClassifier(random_state=42)
        ada_param_grid = {'n_estimators': [100,300], 'learning_rate': [0.1,0.5, 0.6]}
        ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=-1, cv=10, verbose=1)
        ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
        # 排序
        feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                               'importance': ada_grid.best_estimator_.feature_importances_}).sort_values( 'importance', ascending=False)
        features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
        print('---------------------------------------')
        print('Sample 25 Features from Ada Classifier:')
        print(str(features_top_n_ada[:25]))
        
        # ExtraTree
        et_est = ExtraTreesClassifier(random_state=42)
        et_param_grid = {'n_estimators': [100,300], 'min_samples_split': [3, 4], 'max_depth': [15,20]}
        et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
        et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
        # 排序
        feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X), 
                                              'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
        features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
        print('---------------------------------------')
        print('Sample 25 Features from ET Classifier:')
        print(str(features_top_n_et[:25]))

        # 将三个模型挑选出来的前features_top_n_et合并
        features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, 
                                    features_top_n_et], ignore_index=True).drop_duplicates()

        return features_top_n
    
feature_to_pick = 25  # 选择前25个特征
feature_top_n = get_top_n_features(x_train,y,feature_to_pick)
   
x_train = x_train[feature_top_n] 
x_test = x_test[feature_top_n] 
x_train.head()
