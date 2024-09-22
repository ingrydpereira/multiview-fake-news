from enum import Enum
import pandas


class DatasetsNames(Enum):
    LIAR_2 = "Liar_2"
    LIAR_6 = "Liar_6"
    FAKES = "Fakes"
    ISOT = "ISOT"

def get_saved_transformed_data_7_views(dataset_name):
    PATH = "data/data_transformed"

    x_train_tfidf = pandas.read_csv(PATH + "/" + dataset_name + "_x_train_tfidf" + ".csv")
    x_test_tfidf = pandas.read_csv(PATH + "/" + dataset_name + "_x_test_tfidf" + ".csv")

    x_train_count_vec = pandas.read_csv(PATH + "/" + dataset_name + "_x_train_count_vec" + ".csv")
    x_test_count_vec = pandas.read_csv(PATH + "/" + dataset_name + "_x_test_count_vec" + ".csv")

    x_train_w2v = pandas.read_csv(PATH + "/" + dataset_name + "_x_train_w2v" + ".csv")
    x_test_w2v = pandas.read_csv(PATH + "/" + dataset_name + "_x_test_w2v" + ".csv")

    x_train_glove = pandas.read_csv(PATH + "/" + dataset_name + "_x_train_glove" + ".csv")
    x_test_glove = pandas.read_csv(PATH + "/" + dataset_name + "_x_test_glove" + ".csv")

    x_train_fast = pandas.read_csv(PATH + "/" + dataset_name + "_x_train_fast" + ".csv")
    x_test_fast = pandas.read_csv(PATH + "/" + dataset_name + "_x_test_fast" + ".csv")

    x_train_bert = pandas.read_csv(PATH + "/" + dataset_name + "_x_train_bert" + ".csv")
    x_test_bert = pandas.read_csv(PATH + "/" + dataset_name + "_x_test_bert" + ".csv")

    x_train_falcon = pandas.read_csv(PATH + "/" + dataset_name + "_x_train_falcon" + ".csv", header=None)
    x_test_falcon = pandas.read_csv(PATH + "/" + dataset_name + "_x_test_falcon" + ".csv", header=None)

    y_train = pandas.read_csv(PATH + "/" + dataset_name + "_y_train" + ".csv")
    y_test = pandas.read_csv(PATH + "/" + dataset_name + "_y_test" + ".csv")

    key_y = '0'
    if(key_y not in y_train.columns):
        key_y = 'LiarColumnNames.LABEL'

    return x_train_tfidf.values, x_test_tfidf.values, x_train_count_vec.values, x_test_count_vec.values, x_train_w2v.values, x_test_w2v.values, x_train_glove.values, x_test_glove.values, x_train_fast.values, x_test_fast.values, x_train_bert.values, x_test_bert.values, x_train_falcon.values, x_test_falcon.values, y_train[key_y].values, y_test[key_y].values
