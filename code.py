from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn import ensemble

# [height(cm), weight(kg), shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#defining classifiers
clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = naive_bayes.GaussianNB()
clf4 = ensemble.RandomForestClassifier()

#training each classifier
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
clf4 = clf4.fit(X,Y)

#predicting from each classifier
predict1 = clf1.predict([[190,90,40]])
predict2 = clf2.predict([[190,90,40]])
predict3 = clf3.predict([[190,90,40]])
predict4 = clf4.predict([[190,90,40]])

#printing output for each classifier
print("Decision Tree:")
print(predict1)
print("SVM:")
print(predict2)
print("Naive Bayes (Gaussian):")
print(predict3)
print("Random Forest:")
print(predict4)