#对训练集进行数据预处理

#划分数据集
y = train.iloc[:,1]
x_train = train.iloc[:,2:]
x_test = test.iloc[:,1:]

#把性别属性特征二值化
s_train=x_train['Sex'].replace('female',0)
s_train=s_train.replace('male',1)
x_train['Sex']=s_train
#设置性别的哑变量
sex_dummies_df = pd.get_dummies(x_train['Sex'], prefix=x_train[['Sex']].columns[0])
x_train = pd.concat([x_train, sex_dummies_df], axis=1)

#把港口属性特征二值化
s_train=x_train['Embarked'].replace('C',0)
s_train=s_train.replace('Q',1)
s_train=s_train.replace('S',2)
x_train['Embarked']=s_train
。
#设置Embarked的哑变量
emb_dummies_df = pd.get_dummies(x_train['Embarked'], prefix=x_train[['Embarked']].columns[0])
x_train = pd.concat([x_train, emb_dummies_df], axis=1)
x_train.rename(columns = {'Embarked_0.0':'Embarked_0','Embarked_1.0':'Embarked_1', 
                          'Embarked_2.0':'Embarked_2'}, inplace=True)


#把Name的特征根据称呼提取出来
x_train['Title'] = x_train['Name'].str.extract('.+,(.+)').str.extract( '^(.+?)\.').str.strip()
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master'], 'Master'))

x_train['Title'] = x_train['Title'].map(title_Dict)

s_train=x_train['Title'].replace('Officer',0)
s_train=s_train.replace('Mr',1)
s_train=s_train.replace('Miss',2)
s_train=s_train.replace('Mrs',3)
s_train=s_train.replace('Master',4)
s_train=s_train.replace('Royalty',5)
x_train['Title']=s_train
#设置头衔的哑变量
title_dummies_df = pd.get_dummies(x_train['Title'], prefix=x_train[['Title']].columns[0])
x_train = pd.concat([x_train, title_dummies_df], axis=1)

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
x_train['Fare_Category'] = x_train['Fare'].map(fare_category)
#设置费用的哑变量
fare_cat_dummies_df = pd.get_dummies(x_train['Fare_Category'],prefix=x_train[['Fare_Category']].columns[0])
x_train = pd.concat([x_train, fare_cat_dummies_df], axis=1)

#标准化fare属性
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(x_train['Fare'].reshape(-1,1))
x_train['Fare_scaled'] = scaler.fit_transform(x_train['Fare'].reshape(-1,1), fare_scale_param)

#设置Pclass的哑变量
Pclass_dummies_df = pd.get_dummies(x_train['Pclass'],prefix=x_train[['Pclass']].columns[0])
x_train = pd.concat([x_train, Pclass_dummies_df], axis=1)

#设置SibSp的哑变量
SibSp_dummies_df = pd.get_dummies(x_train['SibSp'],prefix=x_train[['SibSp']].columns[0])
x_train = pd.concat([x_train, SibSp_dummies_df], axis=1)

#设置Parch的哑变量
Parch_dummies_df = pd.get_dummies(x_train['Parch'],prefix=x_train[['Parch']].columns[0])
x_train = pd.concat([x_train, Parch_dummies_df], axis=1)
x_train['Parch_9'] = x_train.apply(lambda x: 0, axis=1)

#Ticket
x_train['Ticket_Number'] = x_train['Ticket'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
x_train['Ticket_Number'].fillna(0,inplace=True)

x_train['Ticket_Letter'] = x_train['Ticket'].str.split().str[0]

#设置Ticket_Letter属性
def set_Ticket_Letter(df):
    df.loc[ (df.Ticket_Letter.notnull()), 'Ticket_Letter' ] = "Yes"
    df.loc[ (df.Ticket_Letter.isnull()), 'Ticket_Letter' ] = "No"
    return df
x_train = set_Ticket_Letter(x_train)

s_train=x_train['Ticket_Letter'].replace('No',0)
s_train=s_train.replace('Yes',1)
x_train['Ticket_Letter']=s_train
#设置Ticket_Letter的哑变量
Ticket_Letter_dummies_df = pd.get_dummies(x_train['Ticket_Letter'], prefix=x_train[['Ticket_Letter']].columns[0])
x_train = pd.concat([x_train, Ticket_Letter_dummies_df], axis=1)

#Parch and SibSp这两组数据都能显著影响到Survived，但是影响方式不完全相同，所以将这两项合并成FamilySize组的同时保留这两项。
x_train['Family_Size'] = x_train['Parch'] + x_train['SibSp'] + 1
def family_size_category(Family_Size):
        if Family_Size <= 1:
            return 0
        elif Family_Size <= 5:
            return 1
        else:
            return 2
x_train['Family_Size_Category'] = x_train['Family_Size'].map(family_size_category)

#设置家庭规模的哑变量
family_size_cat_dummies_df = pd.get_dummies(x_train['Family_Size_Category'],prefix=x_train[['Family_Size_Category']].columns[0])
x_train = pd.concat([x_train, family_size_cat_dummies_df], axis=1)

#设置cabin属性
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
x_train = set_Cabin_type(x_train)

s_train=x_train['Cabin'].replace('No',0)
s_train=s_train.replace('Yes',1)
x_train['Cabin']=s_train
#设置cabin的哑变量
Cabin_dummies_df = pd.get_dummies(x_train['Cabin'], prefix=x_train[['Cabin']].columns[0])
x_train = pd.concat([x_train, Cabin_dummies_df], axis=1)

x_train.head()
