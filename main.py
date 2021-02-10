import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from visualize_classifier import visualize_classifier

input_file = 'data_multivar_nb.txt' 

data = np.loadtxt(input_file, delimiter=',') 
#確認
print(data)
X, y = data[:, :-1], data[:, -1] 

classifier = GaussianNB()
classifier.fit(X, y)

y_pred = classifier.predict(X)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0] 
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%") 

visualize_classifier(classifier, X, y)

#訓練用データと検証用データに分ける

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,test_size=0.2, random_state=3) 
classifier_new = GaussianNB() 
classifier_new.fit(X_train, y_train) 
y_test_pred = classifier_new.predict(X_test) 

accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0] 
print("Accuracy of the new classifier =", round(accuracy, 2), "%") 

visualize_classifier(classifier_new, X_test, y_test) 

num_folds = 3 
accuracy_values = model_selection.cross_val_score(classifier, 
                      X, y, scoring='accuracy', cv=num_folds) 
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%") 

precision_values = model_selection.cross_val_score(classifier, 
                      X, y, scoring='precision_weighted', cv=num_folds) 
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%") 

recall_values = model_selection.cross_val_score(classifier, 
                      X, y, scoring='recall_weighted', cv=num_folds) 
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%") 

f1_values = model_selection.cross_val_score(classifier, 
                      X, y, scoring='f1_weighted', cv=num_folds) 
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")


#精度検証

#正解率(Accuracy) TP+TN/全体　　FP,FNを無視していい場合に利用する
#適合率(Precision) TP/TP+FP　　予測値で陽性である判断をされたとき、実際に陽性のサンプルである割合を表す
#再現率(Recall) TP/TP+FN　　　実際に陽性であるサンプルの内、陽性であると判断されたサンプルの割合を表す
#F値(f1-score) 2Precision*recall/precision+recall precisionとrecallの調和平均