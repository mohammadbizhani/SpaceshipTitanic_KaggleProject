import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_prep = train_data.copy()
train_prep.isnull().sum()

test_prep = test_data.copy()
test_prep.isnull().sum()


# Splitting the Cabin fearture and replace 1 total spend instead of 5 float features
train_prep[['Deck','Num','Side']] = train_prep['Cabin'].str.split('/',expand=True)
test_prep[['Deck','Num','Side']] = test_prep['Cabin'].str.split('/',expand=True)


train_final = train_prep.drop(['PassengerId','Cabin','Name','VIP'],axis=1)
test_final = test_prep.drop(['PassengerId','Cabin','Name','VIP'],axis=1)


cols=['HomePlanet','CryoSleep','Destination','Deck','Side', 'Num']
train_final[cols] = train_final[cols].fillna(train_final.mode().iloc[0])
test_final[cols] = test_final[cols].fillna(test_final.mode().iloc[0])

cols2=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
train_final[cols2] = train_final[cols2].fillna(train_final[cols2].mean())
test_final[cols2] = test_final[cols2].fillna(test_final[cols2].mean())


bins=[-1] + list(range(0,1801,300))
labels=list(range(1,len(bins)))

train_final['Cabin_num'] = pd.cut((train_final['Num']).astype(int), bins=bins, labels=labels)
test_final['Cabin_num'] = pd.cut((test_final['Num']).astype(int), bins=bins, labels=labels)


train_final['Total Spend'] = train_final[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] \
                            .sum(axis=1, min_count=1)
test_final['Total Spend'] = test_final[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] \
                            .sum(axis=1, min_count=1)


dummy = ['HomePlanet','CryoSleep','Destination','Cabin_num','Side','Deck']
train_final[dummy] = train_final[dummy].astype('category')
test_final[dummy] = test_final[dummy].astype('category')


Scaler = StandardScaler()
train_final[['Age','Total Spend','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = Scaler.fit_transform \
            (train_final[['Age','Total Spend','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']])
test_final[['Age','Total Spend','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = Scaler.fit_transform \
            (test_final[['Age','Total Spend','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']])


Y = train_final['Transported']
train_final = train_final.drop(['Transported','Num'],axis=1)
X = pd.get_dummies(train_final , columns=['HomePlanet','CryoSleep','Destination','Deck','Side','Cabin_num'], drop_first=True)


test_final = test_final.drop(['Num'],axis=1)
test_final = pd.get_dummies(test_final , columns=['HomePlanet','CryoSleep','Destination','Deck','Side','Cabin_num'], drop_first=True)


X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=1234)



# -----------------------------------------------------------------------------------------------------------------------

RFC = RandomForestClassifier(random_state=1234)


list_cv = {'n_estimators':np.arange(30,60), 'max_depth':np.arange(5,10)}
RFC_CV = GridSearchCV(RFC, list_cv)


RFC_CV.fit(X_train, Y_train)


score_train = RFC_CV.score(X_train, Y_train)
print(RFC_CV.best_params_)
print(RFC_CV.best_score_)


Y_predict = RFC_CV.predict(X_test)
score_test = accuracy_score(Y_test, Y_predict)
print(f"accuracy: {score_test}")


predictions = RFC_CV.predict(test_final)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
output.to_csv('submission8.csv', index=False)


print("Your submission was successfully saved!")

