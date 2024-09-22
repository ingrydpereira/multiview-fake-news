from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from multiviewae import DVCCA, me_mVAE, MoPoEVAE, mvtCAE, wAAE, jointAAE, mVAE
from sklearn.preprocessing import MinMaxScaler
import pandas

from utils.multiview_autoencoders_utils import *
from utils.classifiers_utils import *

random_state = 42
max_epochs = 100
batch_size = 2000

clfs = {"LR": LogisticRegressionCV(random_state=random_state, multi_class='auto', solver='liblinear', penalty='l1'),
        "SVM": SVC(random_state=random_state, kernel='rbf', gamma=1, probability=True),
        "RF": RandomForestClassifier(random_state=random_state, verbose=0, n_estimators=1000, n_jobs=-1),
        "NB": BernoulliNB(),
        "MLP": MLPClassifier(random_state=random_state, batch_size=batch_size, hidden_layer_sizes=(1000,),max_iter=100, activation='relu', solver='adam'),
        "Extra": ExtraTreesClassifier(random_state=random_state, n_estimators=1000, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

models_ae_map = {
    "wAAE": wAAE,
    "jointAAE": jointAAE,
    "mVAE": mVAE,
    "mvtCAE": mvtCAE,
    "DVCCA": DVCCA,
    "me_mVAE": me_mVAE,
    "MoPoEVAE": MoPoEVAE
}

def normalize_data(x_train, x_test):
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)

def get_and_save_results_7_views(models_ae_list,
                    latent_dims,
                    path_result,
                    path_latent,
                    dataset_name,
                    x_train_view1,
                    x_train_view2,
                    x_train_view3,
                    x_train_view4,
                    x_train_view5,
                    x_train_view6,
                    x_train_view7,
                    x_test_view1,
                    x_test_view2,
                    x_test_view3,
                    x_test_view4,
                    x_test_view5,
                    x_test_view6,
                    x_test_view7,
                    y_train,
                    y_test,
                    save_latent = True,
                    save_clfs = True):
    x_train_view1, x_test_view1 = normalize_data(x_train_view1, x_test_view1)
    x_train_view2, x_test_view2 = normalize_data(x_train_view2, x_test_view2)
    x_train_view3, x_test_view3 = normalize_data(x_train_view3, x_test_view3)
    x_train_view4, x_test_view4 = normalize_data(x_train_view4, x_test_view4)
    x_train_view5, x_test_view5 = normalize_data(x_train_view5, x_test_view5)
    x_train_view6, x_test_view6 = normalize_data(x_train_view6, x_test_view6)
    x_train_view7, x_test_view7 = normalize_data(x_train_view7, x_test_view7)

    for model_name in models_ae_list:
        for latent_dim in latent_dims:
            model_ae = train_autoencoder_7_views(models_ae_map[model_name],
                                                 x_train_view1,
                                                 x_train_view2,
                                                 x_train_view3,
                                                 x_train_view4,
                                                 x_train_view5,
                                                 x_train_view6,
                                                 x_train_view7, latent_dim, max_epochs, batch_size)

            train_latent = model_ae.predict_latents(x_train_view1,
                                                     x_train_view2,
                                                     x_train_view3,
                                                     x_train_view4,
                                                     x_train_view5,
                                                     x_train_view6,
                                                     x_train_view7)
            test_latent = model_ae.predict_latents(x_test_view1,
                                                     x_test_view2,
                                                     x_test_view3,
                                                     x_test_view4,
                                                     x_test_view5,
                                                     x_test_view6,
                                                     x_test_view7)

            if save_latent:
                pandas.DataFrame(train_latent[0]).to_csv(path_latent + "/" + dataset_name + "_" + model_name + "_" + str(latent_dim) + "_latent_7views_train" + ".csv")
                pandas.DataFrame(test_latent[0]).to_csv(path_latent + "/" + dataset_name + "_" + model_name + "_" + str(latent_dim) + "_latent_7views_test" + ".csv")

            if save_clfs:
                results = get_results_clfs(clfs, train_latent, test_latent, y_train, y_test, model_name, latent_dim)

                print("Saving results")
                pandas.DataFrame(results,
                                columns=["model AE", "latent dim", "total views", "classifier", "accuracy", "precision",
                                        "recall", "fscore"]).to_csv(path_result + "/" + dataset_name + "_" + model_name + "_" + str(latent_dim) + ".csv")

