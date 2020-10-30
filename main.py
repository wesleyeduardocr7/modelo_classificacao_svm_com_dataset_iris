from sklearn import svm
from sklearn import datasets
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def main():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['label'] = iris.target
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=13)

    clf = svm.SVC(C=1.0)

    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))

    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(X_train, y_train)

    hiperparameters = {
        'kernel': ('linear', 'rbf', 'poly'),
        'C': [1.5, 10, 100, 1000],
        'gamma': [1e-7, 1e-4, 1e-2, 0.1]
    }

    grid = GridSearchCV(svm.SVC(), hiperparameters, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)


main()
