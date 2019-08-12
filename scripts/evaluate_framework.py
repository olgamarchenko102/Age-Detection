import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from time import gmtime, strftime
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import itertools
from matplotlib import pyplot as plt

logger = logging.getLogger()

LABELS = ["18-24", "25-34", "35-49", "50-64", "65-xx"]


def config_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-train_path', required=True, help="path to train data set feature collection")
    parser.add_argument('-test_path', required=True, help="path to test data set feature collection")
    parser.add_argument('-setting_file', required=True, help="path to json file with parameters")
    parser.add_argument('-full_dataset', required=True, help="")
    return parser.parse_args()


# Region "Logging" ================================
def config(file, log_level=logging.INFO):
    logging.basicConfig(level=log_level, format='%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        filename=file, filemode='w')
    formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formatter)
    handler_console.setLevel(log_level)
    logging.getLogger('').addHandler(handler_console)


def init_logging(input_dir, file_name):
    create_dir(input_dir)
    config(file_name, log_level=logging.DEBUG)


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)





def append_to_file(dataset_file_path, text):
    with open(dataset_file_path, "a") as f:
        f.write(text)


def get_scaler(scaler_name):
    logger.info(str.format("Applying scaler:  {0}", scaler_name))
    if scaler_name == "MinMaxScaler":
        return MinMaxScaler(feature_range=(0, 1))
    elif scaler_name == "StandardScaler":
        return StandardScaler(with_mean=True, with_std=True)


def scale(X_train, scaler_name):
    scaler = get_scaler(scaler_name)
    scaler.fit(X_train)
    return scaler.transform(X_train)


def load_classifier(settings):
    if settings["classifier"] == 'knn':
        return KNeighborsClassifier(n_neighbors=settings["n_neighbors"],
                                    leaf_size=settings["leaf_size"],
                                    weights=settings["weights"])


    elif settings["classifier"] == 'svm':
        return SVC(kernel='rbf', C=settings["C"], gamma=settings["gamma"])

    elif settings["classifier"] == 'svm-linear':
        return SVC(kernel='linear', C=settings["C"])

    elif settings["classifier"] == 'random_forest':
        return RandomForestClassifier(min_samples_split=settings["min_samples_split"],
                                      n_estimators=settings["n_estimators"],
                                      max_depth=settings["max_depth"],
                                      max_features=settings["max_features"],
                                      min_samples_leaf=settings["min_samples_leaf"])

    elif settings["classifier"] == 'logistic_regression':
        class_weight = None
        if "class_weight" in settings:
            class_weight = settings["class_weight"]
        return LogisticRegression(multi_class='multinomial',
                                  C=settings["C"],
                                  solver=settings["solver"],
                                  max_iter=settings["max_iter"],
                                  class_weight=class_weight)
    else:
        raise ValueError("Unknown classifier")


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(plot_dir, cm,
                          target_names,
                          title='Confusion matrix of the classifier',
                          cmap=None,
                          normalize=True,
                          plot_name="confusion_matrix.png"):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class\n accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    os.path.expanduser("~/Desktop")
    plt.savefig(os.path.join(plot_dir, plot_name))


def get_genres(author_ids):
    print(arguments.full_dataset)
    genres = []
    with open(arguments.full_dataset) as json_file:
        data = json.load(json_file)
        for author_id in author_ids:
            values_dict = data.get(author_id)
            genre = values_dict.get('genre')
            genres.append(map_genre_name(genre))
    return genres


def get_genre_list(genres_test, genre_id):
    result_list = []
    for item in genres_test:
        if item == genre_id:
            result_list.append(1)
        else:
            result_list.append(0)
    return result_list


def map_genre_name(genre):
    if genre == 1:
        return "twitter"
    if genre == 2:
        return "blogs"
    if genre == 3:
        return "socialmedia"


def create_conf_matrix(project_root_dir, y_test, y_predicted, settings):
    plot_pic_name = str.format("{0}_{1}__conf_matrix.png",
                               settings["language"], settings["classifier"])
    logger.info("Confusion matrix:")
    conf_matrix = confusion_matrix(y_test, y_predicted)

    logger.info(conf_matrix)
    plot_confusion_matrix(project_root_dir, cm=conf_matrix,
                          normalize=False,
                          target_names=LABELS,
                          title=str.format("Confusion matrix of the {0} classifier",settings['classifier']),
                          plot_name=plot_pic_name)


def create_genre_conf_matrix(project_root_dir, y_test, y_predicted, twitts_test, settings, genre):
    genre_indexes = [i for i, x in enumerate(twitts_test) if x == 1]
    y_test_genre = y_test[genre_indexes]
    y_predicted_genre = y_predicted[genre_indexes]

    logger.info("Confusion matrix:")
    conf_matrix = confusion_matrix(y_test_genre, y_predicted_genre)
    logger.info(conf_matrix)
    genre_name = map_genre_name(genre)
    plot_pic_name = str.format("{0}_{1}_{2}__conf_matrix.png",
                               settings["language"], settings["classifier"], genre_name)
    plot_confusion_matrix(project_root_dir, cm=conf_matrix,
                          normalize=False,
                          target_names=LABELS,
                          title=str.format("Confusion matrix of the {0} classifier ({1})",
                                           settings['classifier'], genre_name),
                          plot_name=plot_pic_name)


def get_column_names(file_path):
    f = open(file_path, encoding='utf-8')
    lines = f.readlines()
    first_line = lines[0]
    f.close()
    column_names = []
    items = first_line.split(',')
    for item in items:
        item = item.replace('"', '')
        item = item.replace("'", '')
        column_names.append(item.replace('\n', ''))
    return column_names


def load_feature_collection(train_path, test_path):
    train_feature_names = get_column_names(train_path)[2:]
    test_feature_names = get_column_names(test_path)[2:]


    final_features = ["author_id", "lbl"] + list(set(train_feature_names) & set(test_feature_names))
    print(len(final_features))

    train_full_feature_collection = pd.read_csv(train_path, delimiter=',')
    test_full_feature_collection = pd.read_csv(test_path, delimiter=',')



    final_train_feature_collection = train_full_feature_collection[final_features]
    # final_test_feature_collection = test_full_feature_collection[final_features]
    #
    #print(final_train_feature_collection.shape)


    # final_test_feature_collection = test_full_feature_collection[final_features]
    # print(final_test_feature_collection.shape)

    frame = train_full_feature_collection[final_features[2:]][0:]
    print(frame.shape)

    X_train = np.array(train_full_feature_collection[final_features[2:]][0:])
    y_train = np.array(train_full_feature_collection["lbl"][0:], dtype=np.int)

    X_test = np.array(test_full_feature_collection[final_features[2:]][0:])
    y_test = np.array(test_full_feature_collection["lbl"][0:], dtype=np.int)

    author_ids_test = np.array(test_full_feature_collection["author_id"][0:])

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    return X_train, y_train, X_test, y_test, author_ids_test






if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    plots_dir = os.path.join(project_root_dir, "Confusion_matrices")
    log_dir = os.path.join(project_root_dir, "Log")
    create_dir(project_root_dir)
    create_dir(plots_dir)
    create_dir(log_dir)
    log_file = os.path.join(log_dir, 'evaluation__{0}.log'.format(strftime("%Y-%m-%d__%H-%M-%S", gmtime())))
    init_logging(log_dir, log_file)
    logger.info(str.format("Evaluation started ....."))

    settings_file = arguments.setting_file

    with open(settings_file) as json_file:
        settings = json.load(json_file)

        # 2) Load feature collection
        X_train, y_train, X_test, y_test, author_ids_test =load_feature_collection(arguments.train_path, arguments.test_path)


        print(len(y_test))
        print(len(y_test))

        # 4) Scale data if required
        if settings["scaler_name"] is not None and settings["scaler_name"] != "":
            X_test = scale(X_test, settings["scaler_name"])

        # 5) Normalize data if required
        if settings["normalize"]:
            X_test = normalize(X_test)

        # 6) Select classifier and predict classes
        logger.info("Using classifier: {0}".format(settings["classifier"]))
        classifier = load_classifier(settings)

        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)
        logger.info("Target classes: {}".format(y_test))
        logger.info("Predicted classes: {}".format(y_predicted))

        # 8) Count average f1-score and f1 by class
        f1_scores = f1_score(y_test, y_predicted, average=None)
        f1_mean = np.mean(f1_scores)
        logger.info("Micro avereged F1: {}".format(f1_mean))
        logger.info("F1 scores by class: {}".format(f1_scores))

        # 9) Create and save confusion matrix
        create_conf_matrix(plots_dir, y_test, y_predicted, settings)

        # 10) Create and save confusion matrix by genre
        #genres_test = get_genres(author_ids_test)
        #twitts_test = get_genre_list(genres_test, 1)
        # blogs_test = get_genre_list(genres_test, 2)
        # sm_test = get_genre_list(genres_test, 3)
        #
        #create_genre_conf_matrix(plots_dir, y_test, y_predicted, twitts_test, settings, 1) #twitter
        # create_genre_conf_matrix(plots_dir, y_test, y_predicted, blogs_test, settings, 2) #blogs
        # create_genre_conf_matrix(plots_dir, y_test, y_predicted, sm_test, settings, 3) #sm

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
