#用模型预测训练集 Age 缺失值
missing_age_df = pd.DataFrame(x_train[['Age', 'Parch', 'Sex', 'Cabin','SibSp', 'Family_Size',
                                       'Ticket_Letter','Family_Size_Category','Fare_Category',
                                       'Ticket_Number', 'Title', 'Fare', 'Pclass', 'Embarked']])
    
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]    
    
train_age = missing_age_train.iloc[:,1:]
trainlabel_age = missing_age_train.iloc[:,0]
test_age = missing_age_test.iloc[:,1:]

#这里使用随机森林训练数据
rf_reg = RandomForestRegressor(random_state=2017)
rf_reg_param_grid = {'n_estimators': [30,40,50,60,70,80,90,100], 
                     'max_features': [0.4,0.6,0.8,1,"auto","sqrt" ,"log2"],
                     'min_samples_leaf':[30,40,50]}
rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=5, n_jobs=-1, verbose=1,  scoring='neg_mean_squared_error')
rf_reg_grid.fit(train_age, trainlabel_age)
print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
print('RF Train Error for "Age" Feature Regressor:'+ str(rf_reg_grid.score(train_age, trainlabel_age)))
predictAges = rf_reg_grid.predict(test_age)
x_train.loc[(x_train.Age.isnull()),'Age'] = predictAges

#对于年龄划分等级
def age_size_category(Age):
        if Age <= 12:
            return 0
        elif Age <= 25:
            return 1
        elif Age <= 55:
            return 2
        else:
            return 3       
x_train['age_size_category'] = x_train['Age'].map(age_size_category)
#设置哑变量
age_size_cat_dummies_df = pd.get_dummies(x_train['age_size_category'],prefix=x_train[['age_size_category']].columns[0])
x_train = pd.concat([x_train, age_size_cat_dummies_df], axis=1)

#标准化age属性
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(x_train['Age'].reshape(-1,1))
x_train['Age_scaled'] = scaler.fit_transform(x_train['Age'].reshape(-1,1), age_scale_param)

x_train = x_train.drop(["Ticket","Name",'Age','Fare'],axis=1)

x_train.head(10)
