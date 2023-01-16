from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd  # packages
import numpy as np  # packages
import matplotlib.pyplot as plt  # packages
import sklearn
from sklearn.metrics import precision_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import multilabel_confusion_matrix

from sklearn import cross_validation

y = Personality
droped_list = []  # 16P .csv

TP, FP, TN, FN = 0

min, max = 0


def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return (cm[0][0], cm[0][1], cm[1][0], cm[1][1])


TP, FP, FN, TN = get_confusion_matrix_values(x_test, x_pred)

display = pd.read_csv("16p-Mapping.txt", sep="-", index_col=0)

display.head()
# display1 = pd.read_csv("16P.csv", sep=",")

print(df.head())
# lines = pd.read().splitlines();

print(display)
knn = KNeighborsClassifier(n_neighbors=16)

# x_train, x_test, y_train, y_test = train_test_split(
#     scaled_features, pd['16p-Mapping'], test_size = 0.30)

knn.fit(x_train, y_train)
pred = knn.predict(x_test)

# print(16p-Mapping(y_test, pred))

list = []
for i in range(list):
    print(list[i, :4])

x = display.iloc[:, len(list)].values
y = display.iloc[:, len(list)].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.05, random_state=0)

p16 = []
rows = 0
cols = 0
for i in p16.rows:
    for j in p16.cols:
        if (list[i][j] == 3)
            print("Fully Agree")
        if (list[i][j] == 2)
            print("Partially Agree")
        if (list[i][j] == 1)
            print("Slightly Agree")
        if (list[i][j] == -1)
            print("Slightly DisAgree")
        if (list[i][j] == -2)
            print("Partially DisAgree")
        if (list[i][j] == -3)
            print("Fully DisAgree")
        if (list[i][j] == 0)
            print("Neutral")
data = cross_validation.KFold(len(train_set), n_folds=5, indices=False)

# ----------------------accuracy
print(accuracy_score(y_test, model.predict(x_test)))

# ----------------------precision

precision = precision_score(y_test, pred)

# --------------------recall
recall = recall_score(y_test, pred)
# ----------------------

df = pd.DataFrame(display)
print(df)
scaler = MinMaxScaler()

scaler.fit(x)

normalized_data = scaler.fit_transform(df)
print(normalized_data)

# -------------------normalization
normalized_data = (normalized_data - min) / (max - min)
# -------------------calculate of precision accuracy and recall
precision = TP / (TP + FP)

accuracy = (TP + TN) / (TP + FP + FN + TP)

recall = TP / (TP + FN)

# adı display

actual = display['Actual'].to_numpy()
predicted = display['Predicted'].to_numpy()

conf = multilabel_confusion_matrix(actual, predicted)

print(conf)

TN = conf[:, 0, 0]
TP = conf[:, 1, 1]
FN = conf[:, 1, 0]
FP = conf[:, 0, 1]

print(TN, TP, FN, FP)

# ---------------------------
opened_file = lines.readlines()
# personel sınıflama verileri
son_kelime = opened_file[-4].split(',')[0]

droped_list[1, :]

print(droped_list)

df.drop(index=df.index[0], axis=0, inplace=True)

# read
# droplanacak sütunu düşür
# personality ytes ve ytrain olacak
# train test split
# knn classifier kullan
# predic yap

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

x = numpy.delete(x, (0), axis=0)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

classifier = KNeighborsClassifier(n_neighbors=16, metric='minkowski', p=2)
classifier.fit(X_Train, Y_Train)

Y_Pred = classifier.predict(X_Test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_Test, Y_Train)

# accuracy


train_acc = np.mean(neigh.predict(train.iloc[:, 0:4]) == train.iloc[:, 4])
train_acc

# data frame ve örnek sayıları

digit = load_digits()  # loading the MNIST dataset
dig = pd.DataFrame(digit['data'][:, :])  # Creation of a Panda dataframe
dig.head()  # ilk 5 örneği verir

X_Train = digit.data
Y_Train = digit.target


# euclide hesaplaması

def calc_euclid(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])

    return np.sqrt(distance)


# TN, TP, FN, FP hesaplaması

def analyze(testSet, predictions):
    TN = 0
    TP = 0
    FN = 0
    FP = 0

    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            if testSet[x][-1] == '2':
                TN += 1
            elif testSet[x][-1] == '4':
                TP += 1
        elif testSet[x][-1] != predictions[x]:
            if testSet[x][-1] == '2':
                FN += 1
            elif testSet[x][-1] == '4':
                FP += 1

    # calculate accuracy
    acc = ((TN + TP) / float(len(testSet))) * 100.0

    # calculate TPR, PPV, TNR, and F1 Score
    print('TN =' + repr(TN) + ' TP = ' + repr(TP) + ' FN = ' + repr(FN) + ' FP = ' + repr(FP))
    tpr = (TP / (TP + FN))
    ppv = (TP / (TP + FP))
    tnr = (TN / (TN + FP))
    fs = (2.0 * ppv * tpr) / (ppv + tpr)

    return acc, tpr, ppv, tnr, fs


    # predictors = df.values[:, 0:21]

#  targets = df.values[:,21]


