import seaborn as sns
tips = sns.load_dataset("tips")
print(tips.info())
print(tips.head())

tips_have = tips.iloc[0:220, :]
tips_new = tips.iloc[ 220: , : ]

tips_new.drop(['size'],axis=1, inplace=True)
print(tips_have.shape, tips_new.shape)