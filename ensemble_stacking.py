

'''模型融合中使用到的各个单模型'''
clfs = [AdaBoostClassifier(n_estimators=50,algorithm='SAMME',random_state=2017,learning_rate=0.5 ),
        RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=50,max_features='sqrt',min_samples_split=5,n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=50,max_features='sqrt',min_samples_split=5,n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05,subsample=0.8,n_estimators=100)]

#x_test = x_test.drop(["Ticket_Number"],axis=1)

dataset_blend_train = np.zeros((x_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((x_test.shape[0], len(clfs)))

X = np.array(x_train)
y = np.array(y)
'''5折stacking'''
n_folds = 5
sfk = list(StratifiedKFold(y, n_folds))

for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((x_test.shape[0], len(sfk)))
    for i, (train, test) in enumerate(sfk):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(x_test)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))

#构建xgb矩阵  
xgtrain = xgb.DMatrix(dataset_blend_train,label=y)  

def modelMetrics(clf,X_train,y_train,isCv=True,cv_folds=10,
                 early_stopping_rounds=300):  
    if isCv:  
        xgb_param = clf.get_xgb_params()
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=clf.get_params()['n_estimators'],nfold=cv_folds,  
                          metrics='auc',early_stopping_rounds=early_stopping_rounds) #是否显示目前几颗树
        clf.set_params(n_estimators=cvresult.shape[0])  
  
    #训练
    clf.fit(X_train,y_train,eval_metric='auc')  
  
    #预测  
    train_predictions = clf.predict(X_train)  
    train_predprob = clf.predict_proba(X_train)[:,1]#1的概率 
      #打印  
    print("==================模型报告==================")  
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train,train_predictions))  
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train,train_predprob))  
  
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)  
    feat_imp.plot(kind='bar',title='Feature importance')  
    plt.ylabel('Feature Importance Score')
    plt.show()

    #初始化参数 
xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,
                     colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)     
#1.确定学习速率和tree_based 参数调优的估计器数目
modelMetrics(xgb1,dataset_blend_train,y)

# 网格搜索参数调优
print("==============模型参数调优开始==============")  
print('调优后的决策树数量为：',xgb1.n_estimators )    

#1.learning_rate参数调优  
param_test6 = {  'learning_rate': uniform() }  
gsearch6 = RandomizedSearchCV(
    estimator=XGBClassifier(learning_rate=0.1, n_estimators =xgb1.n_estimators,max_depth=5,
                            min_child_weight=1, gamma=0,subsample=0.8,colsample_bytree=0.8,
                            objective='binary:logistic', nthread=4,scale_pos_weight=1, seed=27),
    param_distributions=param_test6, scoring='roc_auc', n_jobs=-1,iid=False, cv=5,n_iter=200)  
gsearch6.fit(dataset_blend_train,y)  
#gsearch2.grid_scores_
#gsearch2.best_params_, gsearch2.best_score_ 
print('-----------------------------------------')
print('调优后的learning_rate参数为：', gsearch6.best_params_['learning_rate'])
print('本次调优后的最好得分为：',gsearch6.best_score_)          

# 2.max_depth（典型值：3-10）和 min_child_weight 参数调优
param_test1 = {  'max_depth':range(3,10,1),  
                 'min_child_weight':range(0,10,1)  }  
gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate =0.1,n_estimators =xgb1.n_estimators,max_depth=5,
                                                min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,  
                                                objective= 'binary:logistic',nthread=4,scale_pos_weight=1, seed=27),
                        param_grid=param_test1,scoring='roc_auc', n_jobs=-1,iid=False,cv=5)  
gsearch1.fit(dataset_blend_train,y)  
#gsearch1.grid_scores_,
#gsearch1.best_params_,gsearch1.best_score_    
#gsearch1.best_params_['max_depth']
print('-----------------------------------------')
print('调优后的max_depth参数为：', 
      gsearch1.best_params_['max_depth'])
print('调优后的min_child_weight参数为：', 
      gsearch1.best_params_['min_child_weight'])
print('本次调优后的最好得分为：',gsearch1.best_score_)   

#3.gamma参数调优  
param_test2 = {  'gamma': uniform(0,0.5) }  
gsearch2 = RandomizedSearchCV(
    estimator=XGBClassifier(learning_rate=0.1, n_estimators =xgb1.n_estimators,max_depth=gsearch1.best_params_['max_depth'],
                            min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=0,subsample=0.8, 
                            colsample_bytree=0.8,objective='binary:logistic', nthread=4,scale_pos_weight=1, seed=27),
    param_distributions=param_test2, scoring='roc_auc', n_jobs=-1,iid=False, cv=5,n_iter=50)  
gsearch2.fit(dataset_blend_train,y)  
#gsearch2.grid_scores_
#gsearch2.best_params_, gsearch2.best_score_ 
print('-----------------------------------------')
print('调优后的gamma参数为：', gsearch2.best_params_['gamma'])
print('本次调优后的最好得分为：',gsearch2.best_score_)          
    
#4.调整subsample（0.5-1）和 colsample_bytree（0.5-1）参数    
param_test3 = {  'subsample': [i / 20.0 for i in range(10, 20)],  
                 'colsample_bytree': [i / 20.0 for i in range(10, 20)]  }  
 
gsearch3 = GridSearchCV( 
    estimator=XGBClassifier(learning_rate=0.1,n_estimators =xgb1.n_estimators,max_depth=gsearch1.best_params_['max_depth'], 
                            min_child_weight=gsearch1.best_params_['min_child_weight'],gamma=gsearch2.best_params_['gamma'],  
                            subsample=0.8, colsample_bytree=0.8,objective='binary:logistic', nthread=4,  
                            scale_pos_weight=1, seed=27), param_grid=param_test3,
    scoring='roc_auc', n_jobs=-1, iid=False, cv=5)    
gsearch3.fit(dataset_blend_train, y)  
#gsearch3.grid_scores_, 
#gsearch3.best_params_, gsearch3.best_score_ 
print('-----------------------------------------')
print('调优后的subsample参数为：', gsearch3.best_params_['subsample'])
print('调优后的colsample_bytree参数为：', gsearch3.best_params_['colsample_bytree'])
print('本次调优后的最好得分为：',gsearch3.best_score_)    
    
    #5.正则化参数调优   
    #reg_alpha参数调节
param_test4 = {  'reg_alpha': uniform(0,0.1)  }  
gsearch4 = RandomizedSearchCV( estimator=XGBClassifier(learning_rate=0.1,n_estimators =xgb1.n_estimators,
                                                 max_depth=gsearch1.best_params_['max_depth'],
                                                 min_child_weight=gsearch1.best_params_['min_child_weight'],
                                                 gamma=gsearch2.best_params_['gamma'],
                                                 subsample=gsearch3.best_params_['subsample'],
                                                 colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                                                 objective='binary:logistic', nthread=4,scale_pos_weight=1, seed=27),
                        param_distributions=param_test4, scoring='roc_auc',n_jobs=-1,  iid=False, cv=5,n_iter=50)    
gsearch4.fit(dataset_blend_train, y)  
#gsearch4.grid_scores_, 
#gsearch4.best_params_, gsearch4.best_score_  
print('-----------------------------------------')
print('调优后的reg_alpha参数为：', gsearch4.best_params_['reg_alpha'])
print('本次调优后的最好得分为：',gsearch4.best_score_)
    #reg_lambda参数调节 
param_test5 = {  'reg_lambda': uniform(0,1) }  
gsearch5 = RandomizedSearchCV(estimator=XGBClassifier(learning_rate =0.1,n_estimators =xgb1.n_estimators,
                                                      max_depth=gsearch1.best_params_['max_depth'], min_child_weight=gsearch1.best_params_['min_child_weight'],
                                                      gamma=gsearch2.best_params_['gamma'],subsample=gsearch3.best_params_['subsample'],
                                                      colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                                                      reg_alpha=gsearch4.best_params_['reg_alpha'],objective= 'binary:logistic',
                                                      nthread=4,scale_pos_weight=1,seed=27),
                              param_distributions=param_test5,scoring='roc_auc', n_jobs=-1, 
                              iid=False, cv=5,n_iter=100)    
gsearch5.fit(dataset_blend_train, y)  
#gsearch5.grid_scores_,
#gsearch5.best_params_, gsearch5.best_score_  
print('-----------------------------------------')
print('调优后的reg_lambda参数为：', gsearch5.best_params_['reg_lambda'])
print('本次调优后的最好得分为：',gsearch5.best_score_)

# 降低学习速率以及增加决策树
xgb2 = XGBClassifier(learning_rate =gsearch6.best_params_['learning_rate']*0.1,n_estimators =xgb1.n_estimators*1000,max_depth=gsearch1.best_params_['max_depth'], 
                     min_child_weight=gsearch1.best_params_['min_child_weight'],gamma=gsearch2.best_params_['gamma'],
                     subsample=gsearch3.best_params_['subsample'],colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                     reg_alpha=gsearch4.best_params_['reg_alpha'],reg_lambda =gsearch5.best_params_['reg_lambda'],
                     objective= 'binary:logistic',nthread=4, scale_pos_weight=1,seed=27)    
modelMetrics(xgb2,dataset_blend_train,y)

#降低学习速率以及增加决策树
xgb2 = XGBClassifier(
    learning_rate =gsearch6.best_params_['learning_rate'],n_estimators =xgb1.n_estimators*100,max_depth=gsearch1.best_params_['max_depth'],
                     min_child_weight=gsearch1.best_params_['min_child_weight'],gamma=gsearch2.best_params_['gamma'],
                     subsample=gsearch3.best_params_['subsample'],colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                     reg_alpha=gsearch4.best_params_['reg_alpha'],reg_lambda =gsearch5.best_params_['reg_lambda'],
                     objective= 'binary:logistic',nthread=4, scale_pos_weight=1,seed=27)    
modelMetrics(xgb2,dataset_blend_train,y)

# 训练优化模型 
#输入最优参数
params={
'booster':'gbtree',
'objective': 'multi:softmax',
'num_class':2,
'gamma':gsearch2.best_params_['gamma'],
'max_depth':gsearch1.best_params_['max_depth'], 
'lambda':gsearch5.best_params_['reg_lambda'],
'alpha':gsearch4.best_params_['reg_alpha'],
'subsample':gsearch3.best_params_['subsample'], 
'colsample_bytree':gsearch3.best_params_['colsample_bytree'], 
'min_child_weight':gsearch1.best_params_['min_child_weight'], 
'learning_rate':gsearch6.best_params_['learning_rate'],
'silent':1 , 
'seed':27,
'nthread':4,
#'eval_metric': 'auc'
}

plst = list(params.items())
num_rounds = xgb1.n_estimators*1000 # 迭代次数
watchlist = [(xgtrain, 'train')]

print("==================训练优化模型==================")  
model = xgb.train(plst, xgtrain, num_rounds, watchlist,
                  early_stopping_rounds=300)

print ("best_ntree_limit",model.best_ntree_limit )
xgb_test = xgb.DMatrix(dataset_blend_test)
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit )

