import pandas as pd
import random
features = ['CheckAccountStatus', 'DurationInMonth', 'CreditHistory',
            'Purpose', 'CreditAmount', 'SavingsAccount', 'EmploymentSince',
            'InstallmentRate', 'PersonalStatus', 'OtherDebtors',
            'ResidenceSince', 'Property', 'Age', 'OtherInstallmentPlan',
            'Housing', 'NumberOfCredit', 'Job', 'LiablePersonNumber',
            'Telephone', 'ForeignWorker', 'Credit']
category = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]    # index of categorized feature
test_num = 100
f = open('german.data', 'r')
german_data = pd.read_csv(f, delimiter=' ', names=features)

# 1 : Good, 0 : Bad
answer = german_data['Credit'].replace([1, 2], [1, 0])
german_data['Credit'] = answer

row, col = german_data.shape
for r in range(row):
    for c in category:
        val = german_data.ix[r, features[c]]
        # Attribute from 1 to 9 (A + 1digit)
        if c <= 8:
            german_data.ix[r, features[c]] = str(val)[2:]
        # Attribute from 10 to 20 (A + 2digit)
        else:
            german_data.ix[r, features[c]] = str(val)[3:]

test_i = random.sample(range(0, 1000), test_num)
train_i = [i for i in range(0, 1000) if i not in test_i]
test = pd.DataFrame(german_data.iloc[test_i])
train = pd.DataFrame(german_data.iloc[train_i])

german_data.to_csv('statlog.csv')
train.to_csv('statlog_train.csv')
test.to_csv('statlog_test.csv')

f.close()
