# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:50:24 2023

@author: Mohamad
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Read Train and Test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# make a copy from datas and make changes on them
train_prep = train_data.copy()
test_prep = test_data.copy()


# Distribution for Homeplanet feature
list_homeplanet = Counter(list(train_data['HomePlanet']))

name_homeplanet = list(list_homeplanet.keys())
number_homeplanet = list(list_homeplanet.values())

plt.figure('Bar chart for HomePlanet')
plt.pie(number_homeplanet, labels=name_homeplanet, autopct='%.1f%%', explode=[0,0,0,0.1])
plt.title('HomePlanet of the passengers', fontweight='bold')
plt.show()


# Transported by Homeplanet
plt.figure('Transported people by Homeplanet')
sns.set_style('ticks')
sns.countplot(data=train_data, x='HomePlanet', hue='Transported', palette='YlGnBu')
plt.title('Transported By HomePlanet', fontweight='bold')
plt.show()


# Transported by CryoSleep
sns.set_style('ticks')
plt.figure('Transported people by CryoSleep')
sns.countplot(data=train_data, x='Transported', hue='CryoSleep', palette='Set2')
plt.title('Transported By CryoSleep', fontweight='bold')
plt.show()


# Transported by Destination
sns.set_style('dark')
plt.figure('Transported people by Destination')
sns.countplot(data=train_data, x='Destination',hue='Transported',palette="YlOrBr")
plt.title('Transported By Destination', fontweight='bold')
plt.show()


# Transported by VIP
sns.set_style('ticks')
plt.figure('Transported people by VIP')
sns.countplot(data=train_data, x='Transported',hue='VIP',palette="vlag")
plt.title('Transported By VIP', fontweight='bold')
plt.show()


# Splitting the Cabin fearture and replace 1 total spend instead of 5 float features
train_prep[['Deck','Num','Side']] = train_prep['Cabin'].str.split('/',expand=True)

train_prep['Total Spend'] = train_prep[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] \
                            .sum(axis=1, min_count=1)


# Transported by Deck
deck_feature = train_prep.dropna(subset=['Deck'])
orderd = sorted(deck_feature['Deck'].unique())                             
sns.set_style('ticks')
plt.figure('Transported people by Deck')
sns.countplot(data=train_prep, x='Deck',hue='Transported',palette='magma', order=orderd)
plt.title('Transported By Deck', fontweight='bold')
plt.legend(loc='upper left')
plt.show()


# Transported by Side
sns.set_style('ticks')
plt.figure('Transported people by Side')
sns.countplot(data=train_prep, x='Side',hue='Transported',palette="viridis")
plt.title('Transported By Side', fontweight='bold')
plt.show()


# Boxplot for Transported by Age
plt.figure('Boxplot Transported by Age')
plt.title('Boxplot for Age')
sns.boxplot(x='Transported', y='Age', data=train_prep, palette="mako")
plt.show()


# heatmap for train_data
plt.figure("Heatmap for correlations")
corr = train_prep[['Age','Total Spend','Transported']].corr()
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.title('Heatmap for numeric features', fontweight='bold')
plt.tight_layout()
plt.show()


# Distribution for Age and Total Spend 
train_prep[['Age','Total Spend']].hist(rwidth=0.9)
plt.tight_layout()
plt.show()












