#test = pd.read_excel('E:/数据集/Titanic/test.xlsx')
x_test = test.iloc[:,1:]
#同样的，对测试集进行数据预处理
s_test=x_test['Sex'].replace('female',0)
s_test=s_test.replace('male',1)
x_test['Sex']=s_test
#设置性别的哑变量
sex_dummies_df = pd.get_dummies(x_test['Sex'], prefix=x_test[['Sex']].columns[0])
x_test = pd.concat([x_test, sex_dummies_df], axis=1)

s_test=x_test['Embarked'].replace('C',0)
s_test=s_test.replace('Q',1)
s_test=s_test.replace('S',2)
x_test['Embarked']=s_test
#设置登录港口的哑变量
emb_dummies_df = pd.get_dummies(x_test['Embarked'],prefix=x_test[['Embarked']].columns[0])
x_test = pd.concat([x_test, emb_dummies_df], axis=1)

#把Name的特征根据称呼提取出来
x_test['Title'] = x_test['Name'].str.extract('.+,(.+)').str.extract( '^(.+?)\.').str.strip()

title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master'], 'Master'))

x_test['Title'] = x_test['Title'].map(title_Dict)

s_test=x_test['Title'].replace('Officer',0)
s_test=s_test.replace('Mr',1)
s_test=s_test.replace('Miss',2)
s_test=s_test.replace('Mrs',3)
s_test=s_test.replace('Master',4)
s_test=s_test.replace('Royalty',5)
x_test['Title']=s_test
#设置头衔的哑变量
title_dummies_df = pd.get_dummies(x_test['Title'], prefix=x_test[['Title']].columns[0])
x_test = pd.concat([x_test, title_dummies_df], axis=1)

#设置Pclass的哑变量
Pclass_dummies_df = pd.get_dummies(x_test['Pclass'],prefix=x_test[['Pclass']].columns[0])
x_test = pd.concat([x_test, Pclass_dummies_df], axis=1)

#设置SibSp的哑变量
SibSp_dummies_df = pd.get_dummies(x_test['SibSp'],prefix=x_test[['SibSp']].columns[0])
x_test = pd.concat([x_test, SibSp_dummies_df], axis=1)

#设置Parch的哑变量
Parch_dummies_df = pd.get_dummies(x_test['Parch'],prefix=x_test[['Parch']].columns[0])
x_test = pd.concat([x_test, Parch_dummies_df], axis=1)

#Ticket
x_test['Ticket_Number'] = x_test['Ticket'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
x_test['Ticket_Number'].fillna(0,inplace=True)
x_test['Ticket_Letter'] = x_test['Ticket'].str.split().str[0]

#设置Ticket_Letter属性
def set_Ticket_Letter(df):
    df.loc[ (df.Ticket_Letter.notnull()), 'Ticket_Letter' ] = "Yes"
    df.loc[ (df.Ticket_Letter.isnull()), 'Ticket_Letter' ] = "No"
    return df
x_test = set_Ticket_Letter(x_test)

s_test=x_test['Ticket_Letter'].replace('No',0)
s_test=s_test.replace('Yes',1)
x_test['Ticket_Letter']=s_test
#设置Ticket_Letter的哑变量
Ticket_Letter_dummies_df = pd.get_dummies(x_test['Ticket_Letter'], prefix=x_test[['Ticket_Letter']].columns[0])
x_test = pd.concat([x_test, Ticket_Letter_dummies_df], axis=1)


#测试集的fare有一个缺失值，用均值填补
x_test.Fare[x_test.Fare.isnull()] = x_test.Fare.dropna().mean()
#把费用划分等级
def fare_category(fare):
        if fare <= 4:
            return 0
        elif fare <= 15:
            return 1
        elif fare <= 25:
            return 2
        elif fare <= 35:
            return 3
        else:
            return 4
x_test['Fare_Category'] = x_test['Fare'].map(fare_category)
#设置费用的哑变量
fare_cat_dummies_df = pd.get_dummies(x_test['Fare_Category'],prefix=x_test[['Fare_Category']].columns[0])
x_test = pd.concat([x_test, fare_cat_dummies_df], axis=1)

#标准化fare属性
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(x_test['Fare'].reshape(-1,1))
x_test['Fare_scaled'] = scaler.fit_transform(x_test['Fare'].reshape(-1,1), fare_scale_param)


#Parch and SibSp这两组数据都能显著影响到Survived，但是影响方式不完全相同，所以将这两项合并成FamilySize组的同时保留这两项。
x_test['Family_Size'] = x_test['Parch'] + x_test['SibSp'] + 1

def family_size_category(Family_Size):
        if Family_Size <= 1:
            return 0
        elif Family_Size <= 5:
            return 1
        else:
            return 2

x_test['Family_Size_Category'] = x_test['Family_Size'].map(family_size_category)
#设置家庭规模的哑变量
family_size_cat_dummies_df = pd.get_dummies(x_test['Family_Size_Category'],prefix=x_test[['Family_Size_Category']].columns[0])
x_test = pd.concat([x_test, family_size_cat_dummies_df], axis=1)

#设置cabin属性
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
x_test = set_Cabin_type(x_test)

s_test=x_test['Cabin'].replace('No',0)
s_test=s_test.replace('Yes',1)
x_test['Cabin']=s_test
#设置cabin的哑变量
Cabin_dummies_df = pd.get_dummies(x_test['Cabin'], prefix=x_test[['Cabin']].columns[0])
x_test = pd.concat([x_test, Cabin_dummies_df], axis=1)

x_test.head()

#用模型预测测试集 Age 缺失值
missing_age_df = pd.DataFrame(x_test[['Age', 'Parch', 'Sex', 'SibSp', 'Family_Size','Cabin',
                                      'Family_Size_Category','Fare_Category','Ticket_Number',
                                       'Ticket_Letter','Title', 'Fare', 'Pclass', 'Embarked']])
    
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]    
    
train_age = missing_age_train.iloc[:,1:]
trainlabel_age = missing_age_train.iloc[:,0]
test_age = missing_age_test.iloc[:,1:]

#这里使用随机森林训练数据

rf_reg = RandomForestRegressor(random_state=42)

rf_reg_param_grid = {'n_estimators': [30,40,50,60,70,80,90,100], 
                     'max_features': [0.4,0.6,0.8,1, "auto","sqrt" ,"log2"],
                     'min_samples_leaf':[30,40,50]}

rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=5, n_jobs=-1, verbose=1,  scoring='neg_mean_squared_error')
rf_reg_grid.fit(train_age, trainlabel_age)
print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
print('RF Train Error for "Age" Feature Regressor:'+ str(rf_reg_grid.score(train_age, trainlabel_age)))
predictAges = rf_reg_grid.predict(test_age)
x_test.loc[(x_test.Age.isnull()),'Age'] = predictAges

#rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
#rfr.fit(train_age,trainlabel_age)
#predictAges = rfr.predict(test_age)
#x_test.loc[(x_test.Age.isnull()),'Age'] = predictAges

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
x_test['age_size_category'] = x_test['Age'].map(age_size_category)
#设置哑变量
age_size_cat_dummies_df = pd.get_dummies(x_test['age_size_category'],
                                         prefix=x_test[['age_size_category']].columns[0])
x_test = pd.concat([x_test, age_size_cat_dummies_df], axis=1)

#标准化age属性
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(x_test['Age'].reshape(-1,1))
x_test['Age_scaled'] = scaler.fit_transform(x_test['Age'].reshape(-1,1), age_scale_param)


x_test = x_test.drop(["Ticket","Name",'Age','Fare'],axis=1)

x_test.head(10)
