import os
import sys
import json
import time
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from time import gmtime, strftime
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger()
LABELS = ["18-24", "25-34", "35-49", "50-64", "65-xx"]


def config_arg_parser():
    """
    Set Parameters to argument parser
    :return: parser arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-train_path', required=True, help="path to train data set feature collection")
    parser.add_argument('-test_path', required=True, help="path to test data set feature collection")
    parser.add_argument('-setting_file', required=True, help="path to json file with hyperparameters for classificators")
    return parser.parse_args()


def config(file, log_level=logging.INFO):
    """
    configure logging and logging message format
    :param file: log file name
    :param log_level:  logging level
    """
    logging.basicConfig(level=log_level, format='%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        filename=file, filemode='w')
    formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formatter)
    handler_console.setLevel(log_level)
    logging.getLogger('').addHandler(handler_console)


def init_logging(input_dir, file_name):
    """
    Create Log directory and set log level
    :param input_dir: root log dir
    :param file_name: log file name
    """
    create_dir(input_dir)
    config(file_name, log_level=logging.DEBUG)


def create_dir(dir_name):
    """
    Create directory
    :param dir_path: directory path
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def append_to_file(dataset_file_path, text):
    """
    Write text to the textfile
    :param dataset_file_path: destination file path
    :param text: text
    """
    with open(dataset_file_path, "a") as f:
        f.write(text)


def get_scaler(scaler_name):
    """
    Get Scaler with its parameters
    :param scaler_name: name of the scaler
    :return: Scaler with setted parameters
    """
    logger.info(str.format("Applying scaler:  {0}", scaler_name))
    if scaler_name == "MinMaxScaler":
        return MinMaxScaler(feature_range=(0, 1))
    elif scaler_name == "StandardScaler":
        return StandardScaler(with_mean=True, with_std=True)


def scale(X_train, scaler_name):
    """
    Scale/normalize the input data set
    :param X_train: data set values
    :param scaler_name: name of the scaler
    :return: transformed data set
    """
    scaler = get_scaler(scaler_name)
    scaler.fit(X_train)
    return scaler.transform(X_train)


def load_classifier(settings):
    """
    Load classifier function with its parameter
    :param settings: dictionary with setting parameters
    :return: classification function with appropriate parameters
    """
    if settings["classifier"] == 'knn':
        return KNeighborsClassifier(n_neighbors=settings["n_neighbors"],
                                    leaf_size=settings["leaf_size"],
                                    weights=settings["weights"])


    elif settings["classifier"] == 'svm':
        return SVC(kernel='rbf', C=settings["C"], gamma=settings["gamma"])

    elif settings["classifier"] == 'svm-linear':
        return SVC(kernel='linear', C=settings["C"])

    elif settings["classifier"] == 'random_forest':
        return RandomForestClassifier(n_estimators=settings["n_estimators"],
                                      max_depth=settings["max_depth"])

    elif settings["classifier"] == 'logistic_regression':

        return LogisticRegression(multi_class='multinomial',
                                  penalty='l2',
                                  C=settings["C"],
                                  solver="lbfgs",
                                  max_iter=20,
                                  class_weight=None)
    else:
        raise ValueError("Unknown classifier")


def plot_confusion_matrix(plot_dir, cm,
                          target_names,
                          title='Confusion matrix of the classifier',
                          cmap=None,
                          normalize=True,
                          plot_name="confusion_matrix.png"):
    """
    Plot confusion matrix. The code of this function is based on
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    :param plot_dir: directory path to save plot graphic
    :param cm: values of the confusion matrix
    :param target_names:  label names
    :param title: title of graphic
    :param cmap: color setting
    :param normalize: normalise
    :param plot_name: name of the final file
    """
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


def create_conf_matrix(project_root_dir, y_test, y_predicted, settings):
    """
    Create confusion matrix figure
    :param project_root_dir: directory path to save figure
    :param y_test: np.array of class labels
    :param y_predicted: np.array of data values
    :param settings: dictionary with settings
    """
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


def get_column_names(file_path):
    """
    Get column names from the data frame
    :param file_path: file path to the feature collection data frame
    :return: column names
    """
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
    """
    Load documents' feature values and responsing lass labels
    :param train_path: path to the training documents' feature collection
    :param test_path: path to the test documents' feature collection
    :return: np.array of test and train feature values,
            and np.array of the test and train class labels
    """
    train_feature_names = get_column_names(train_path)[2:]
    test_feature_names = get_column_names(test_path)[2:]

    final_features = list(set(train_feature_names) & set(test_feature_names))
    logger.info(str.format("Number of common features: {0}", len(final_features)))
    train_full_feature_collection = pd.read_csv(train_path, delimiter=',')
    test_full_feature_collection = pd.read_csv(test_path, delimiter=',')

    X_train = np.array(train_full_feature_collection[final_features])
    y_train = np.array(train_full_feature_collection["lbl"])
    X_test = np.array(test_full_feature_collection[final_features])
    y_test = np.array(test_full_feature_collection["lbl"])

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    # 1) Create directories to save and log the evaluation results
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
        X_train, y_train, X_test, y_test =load_feature_collection(arguments.train_path, arguments.test_path)
        logger.info(str.format("Size of training data set: {0}", X_train.shape))
        logger.info(str.format("Size of test data set: {0}", X_test.shape))

        # 4) Scale data if required
        if settings["scaler_name"] is not None and settings["scaler_name"] != "":
            X_test = scale(X_test, settings["scaler_name"])

        # 5) Normalize data if required
        if settings["normalize"]:
            X_test = normalize(X_test)

        # 6) Select classifier and predict classes
        logger.info("Using classifier: {0}".format(settings["classifier"]))
        classifier = load_classifier(settings)

        # 7) Fit classifier and predict labels for the trai dataset
        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)

        # 8) Count average f1-score and f1 by class
        f1_scores = f1_score(y_test, y_predicted, average=None)
        f1_mean = np.mean(f1_scores)
        logger.info("Micro avereged F1: {}".format(f1_mean))
        logger.info("F1 scores by class: {}".format(f1_scores))

        # 9) Create and save confusion matrix
        create_conf_matrix(plots_dir, y_test, y_predicted, settings)

    # 10) Log running time
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))