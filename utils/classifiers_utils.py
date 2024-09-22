import numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_models(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)

    return accuracy_score(y_test, result), \
           precision_score(y_test, result, average='macro'), \
           recall_score(y_test, result, average='macro'), \
           f1_score(y_test, result, average='macro')


def get_results_clfs(clfs, train_latent, test_latent, y_train, y_test, model_name, latent_dim):
    if len(train_latent) == 1:
        x_train = train_latent[0]
        x_test = test_latent[0]
    else:
        x_train = numpy.concatenate(train_latent, axis=1)
        x_test = numpy.concatenate(test_latent, axis=1)

    results = []

    for clf_name, clf in clfs.items():
        print("Train " + clf_name)
        accuracy, precision, recall, fscore = train_models(clf, x_train, y_train, x_test, y_test)
        results.append([model_name, str(latent_dim), str(len(train_latent)), clf_name, accuracy, precision, recall, fscore])

    return results
