

'''模型融合中使用到的各个单模型'''
clfs = [AdaBoostClassifier(n_estimators=50,algorithm='SAMME',random_state=2017,learning_rate=0.5 ),
        RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=50,max_features='sqrt',min_samples_split=5,n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=50,max_features='sqrt',min_samples_split=5,n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05,subsample=0.8,n_estimators=100)]
