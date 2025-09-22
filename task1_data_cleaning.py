import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_theme(style="whitegrid")
os.makedirs("images", exist_ok=True)

df = pd.read_csv("titanic.csv")
print("Initial Data Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.histplot(df['Age'], kde=True, bins=30, color="skyblue")
plt.title("Age Distribution (Before Cleaning)")
plt.xlabel("Age")
plt.subplot(1,2,2)
sns.boxplot(x=df['Fare'], color="salmon")
plt.title("Fare Outliers (Before Cleaning)")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("images/before_cleaning.png")
plt.close()

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

plt.figure(figsize=(12,5))
sns.boxplot(x=df['Fare'], color="lightgreen")
plt.title("Fare Outliers (Before Removal - Standardized)")
plt.xlabel("Standardized Fare")
plt.savefig("images/fare_outliers.png")
plt.close()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.histplot(df['Age'], kde=True, bins=30, color="orange")
plt.title("Age Distribution (After Cleaning & Scaling)")
plt.xlabel("Standardized Age")
plt.subplot(1,2,2)
sns.boxplot(x=df['Fare'], color="purple")
plt.title("Fare Outliers (After Removal & Scaling)")
plt.xlabel("Standardized Fare")
plt.tight_layout()
plt.savefig("images/after_cleaning.png")
plt.close()

plt.figure(figsize=(7,5))
sns.countplot(x="Sex", hue="Survived", data=df, palette="Set2")
plt.title("Survival Count by Sex (After Cleaning)")
plt.xlabel("Sex (0=Female, 1=Male)")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.savefig("images/survival_by_sex.png")
plt.close()

print("\nCleaned Data Info:")
print(df.info())
print("\nSample Data:\n", df.head())