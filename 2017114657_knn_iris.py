from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
iris = load_iris()

print(iris)
print(iris.data)

print(iris.feature_names)

print(iris.target)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

print(X_train.shape)
print(X_test.shape)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

scores = metrics.accuracy_score(y_test, y_pred)
print(scores)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

classes = {0:'setosa', 1:'versicolor', 2:'virginica'}

x_new =[[3,4,5,2],[5,4,2,2]]
y_predict = knn.predict(x_new)
print(y_predict)

print(classes[y_predict[0]])
print(classes[y_predict[1]])