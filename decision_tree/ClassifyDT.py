def classify(features_train, labels_train):   
   
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    return clf.fit(features_train, labels_train)
    
def DTAccuracy(features_train, labels_train, features_test, labels_test):
    
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy = clf.score(features_test, labels_test)
    return accuracy   