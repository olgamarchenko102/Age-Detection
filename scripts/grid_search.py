import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from operator import itemgetter
from time import gmtime, strftime
from sklearn.pipeline import Pipeline
from ReliefF import ReliefF as reliefF
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
warnings.filterwarnings('ignore')

logger = logging.getLogger()


def config_arg_parser():
    """
    Set Parameters to argument parser
    :return: parser arguments
    """
    parser = argparse.ArgumentParser(description='Grid Search')
    parser.add_argument('-train_path', required=True, help="Train feature collection path")
    parser.add_argument('-language', required=True, help='Tanguage of the dataset')
    parser.add_argument('-classifier', required=True, help='Classifier name, e.g. knn, svm-linear, '
                                                           'svm, random_forest, logistic_regression')
    parser.add_argument('-cross_val', required=True, type=int,
                        help='n-fold crossvalidation value', default=4)
    parser.add_argument('-scaler_name', required=False, help='Scaler name')
    parser.add_argument('-normalize', required=True, default=True, help='Normalise feature values, '
                                                                        'e.g. True r False')
    parser.add_argument('-feature_selector', required=False, default='relieff',  help='Name of the feature selector method')
    parser.add_argument('-kBest__n_features', required=False, type=int, default=3, help='Number of best features to select')
    parser.add_argument('-variance_thr', type=float, required=False, help='Value for variance threshold', default=0.0) #0.8*(1- 0.8)
    parser.add_argument('-relieff__n_features_to_keep', required=False, type=int, default=1, help='Number of features to select')
    parser.add_argument('-relieff__n_neighbors', type=int, required=False, help='Number of neighbors for relieff', default=3)
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


def load_feature_collection(csv_path):
    """
    Load feature values from csv file
    :param csv_path: path to the saved feature values
    :return: np.array of the feature values, and np.array of the class labels,
           list of the feature names, list of the author ids
    """
    feature_names = get_column_names(csv_path)
    feature_collection = pd.read_csv(csv_path, delimiter=',')
    data = np.array(feature_collection[feature_names[2:]][0:])  # , dtype=np.float
    categories = np.array(feature_collection["lbl"][0:], dtype=np.int)
    author_ids = np.array(feature_collection["author_id"][0:])
    header_row = np.array(feature_names)
    features = header_row[2:]
    return data, categories, features, author_ids


def get_scaler(scaler_name):
    """
    Get Scaler with its parameters
    :param scaler_name: name of the scaler
    :return: Scaler with setted parameters
    """
    print("Applying ", scaler_name)
    if scaler_name == "MinMaxScaler":
        return MinMaxScaler(feature_range=(0, 1))
    elif scaler_name == "StandardScaler":
        return StandardScaler(with_mean=True, with_std=True)


def scale_train_test(X_train, X_test, scaler_name):
    """
    Scale the input data sets
    :param X_train: values of train data set
    :param X_test: values of test data set
    :param scaler_name: name of the scaler
    :return: transformed data sets
    """
    scaler = get_scaler(scaler_name)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def scale(X_train, scaler_name):
    """
    Scale/normalize the input data set
    :param X_train: data set values
    :param scaler_name: name of the scaler
    :return: transformed data set
    """
    scaler = get_scaler(scaler_name)
    return scaler.fit_transform(X_train)


def knn_build(n_neighbors=5, leaf_size=30, weights='uniform'):
    """

    :param n_neighbors: number of neighbors
    :param leaf_size: number of objects in leaf
    :param weights: weight function
    :return: knn classifier function
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, weights=weights)


def knn_param_grid():
    """
    Set parameters for knn parameter grid
    :return: parameter grid
    """
    return {
        "classifier__n_neighbors": [5, 7, 9, 11, 15, 17, 21, 23, 25, 30, 35],
        "classifier__leaf_size": [5, 10, 15, 20, 25, 30 ],
        "classifier__weights": ["uniform"]
    }


def svm_build(kernel, random_state=0, C=1.0, gamma='auto'):
    """
    Get svm classifier function
    :param kernel: kernel function
    :param random_state:
    :param gamma: variance value
    :return: svm classifier function
    """
    return SVC(kernel=kernel, class_weight='balanced', random_state=random_state, C=C, gamma=gamma)


def svm_param_grid_rbf():
    """
    Set parameters for svm_rgb parameter grid
    :return: parameter grid
    """
    c_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_params = [0.001, 0.01, 0.1, 1, 10, 100]
    return {'classifier__C': c_params,
            'classifier__gamma': gamma_params}


def svm_param_grid_linear():
    """
    Set parameters for svm classifier
    :return:
    """
    c_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    return {'classifier__C': c_params}


def random_forest_build(random_state=0, n_jobs=1, min_samples_split=2, n_estimators=100, max_depth=None, max_features='auto',
          min_samples_leaf=1, bootstrap=True, criterion='gini'):
    """
    Get random forest classifier function
    :param random_state: random state value
    :param n_jobs: number of n_jobs
    :param min_samples_split: min number of sample splits
    :param n_estimators: number of estimators(trees)
    :param max_depth: maximal depth of tree
    :param max_features: maximal number of features
    :param min_samples_leaf: mininal number of  elements in tree leaf
    :param bootstrap: bootstapping
    :param criterion: accuracy criterion measure
    :return: random forest classifier function
    """
    return RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=n_jobs,
                                  min_samples_split=min_samples_split, n_estimators=n_estimators, max_depth=max_depth,
                                  max_features=max_features, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                                  criterion=criterion)


def random_forest_param_grid():
    """
    Set parameters for logistic regression parameter grid
    :return: parameter grid
    """
    return {"classifier__n_estimators": [5, 10, 15, 20, 25, 50],
            "classifier__max_depth": [5, 25, 50, 100]}


def logistic_regression_param_grid():
    """
    Set parameters for logistic regression parameter grid
    :return: parameter grid
    """
    return {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": ["balanced", None]
            }


def logistic_regression_build(penalty='l2', multi_class='multinomial', C=10, class_weight="balanced", solver="lbfgs",max_iter=20):
    """
    Get logistic regression classifier function
    :param penalty: penalty function
    :param multi_class: class problem
    :param C: C value
    :param class_weight: weight type
    :param solver: solver name
    :param max_iter: max number of iterations
    :return: logistic regression classifier function
    """
    return LogisticRegression(penalty=penalty, multi_class=multi_class,
                              C=C, class_weight=class_weight, solver=solver, max_iter=max_iter)


def get_classifier(clf_name):
    """
    Get classifier ant its parameter grid
    :param clf_name:  classifier name
    :return: classifier function and  its parameter grid
    """
    if clf_name == 'svm':
        return svm_build(kernel='rbf'), svm_param_grid_rbf()
    elif clf_name == 'svm-linear':
        return svm_build(kernel='linear'), svm_param_grid_linear()
    elif clf_name == 'knn':
        return knn_build(), knn_param_grid()
    elif clf_name == 'random_forest':
        return random_forest_build(), random_forest_param_grid()
    elif clf_name == 'logistic_regression':
        return logistic_regression_build(), logistic_regression_param_grid()
    else:
        raise ValueError("Unknown classifier")


def get_feature_subset(X, y, features, n_neighbors, relieff__n_features_to_keep):
    """
    Get subset of feature values
    :param X: feature values
    :param y: class labeös
    :param features: all features
    :param n_neighbors: number if neighbors
    :param relieff__n_features_to_keep: number f features to select
    :return: subset of feature values
    """
    relief = reliefF(n_neighbors=n_neighbors, n_features_to_keep=relieff__n_features_to_keep)
    X_subset = relief.fit_transform(X,y)
    feature_indexes = relief.top_features[:relieff__n_features_to_keep]
    selected_features = features[feature_indexes]
    return X_subset, y, selected_features


def k_best_build():
    """
    Build SelectKBest function
    :return: SelectKBest selector function
    """
    return SelectKBest()


def k_best_param_grid(kBest_n_features):
    """
    Get parameters for SelectKBest selector
    :param kBest_n_features: number of features to select
    :return: SelectKBest parameter grid
    """
    return {"feature_selector__score_func": [f_classif],
            "feature_selector__k": [kBest_n_features]}


def variance_threshold_build():
    """
    Build threshold function
    :return: Variance threshold selector function
    """
    return VarianceThreshold()


def variance_threshold_param_grid(variance_thr):
    """
    Get parameters for variance threshold
    :param variance_thr: variance threshold value
    :return: variance threshod parameter grid
    """
    return {"feature_selector__threshold": [variance_thr]}


def get_selector(selector_name, kBest_n_features, variance_thr):
    """
    Get the feature selector and its parameters
    :param selector_name: name of selector
    :param kBest_n_features: number of features to select
    :param variance_thr:  variance threshold value
    :return: selector and its parameters
    """
    if selector_name.lower() == "k_best":
        return k_best_build(), k_best_param_grid(kBest_n_features=kBest_n_features)
    elif selector_name.lower() == "variance_threshold":
        return variance_threshold_build(), variance_threshold_param_grid(variance_thr=variance_thr)
    else:
        raise ValueError("Unknown selector")


def build_pipeline():
    """Bild the processing pipeline"""
    if arguments.feature_selector.lower() == "relieff":
        return Pipeline(steps=[('classifier', classifier)])
    else:
        if arguments.scaler_name is None:
            return Pipeline(steps=[('feature_selector', feature_selector),
                                   ('classifier', classifier)
                                   ])
        elif arguments.scaler_name == "MinMaxScaler":
            return Pipeline(steps=[('scaler', MinMaxScaler(feature_range=(0, 1))),
                                   ('feature_selector', feature_selector),
                                   ('classifier', classifier)
                                   ])
        elif arguments.scaler_name == "StandardScaler":
            return Pipeline(steps=[('scaler', StandardScaler(with_mean=True, with_std=True)),
                                   ('feature_selector', feature_selector),
                                   ('classifier', classifier)
                                   ])


def create_dir(dir_name):
    """
    Create Log directory and set log level
    :param input_dir: root log dir
    :param file_name: log file name
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def check_input_params():
    """
    Check if language parameters are correct
    """
    if arguments.language.lower() not in ['english', 'spanish']:
        logger.info('ERROR: Wrong language input. Try English or Spanish')
        sys.exit(1)


def append_to_file(dataset_file_path, text):
    """
    Write text to file
    :param dataset_file_path: file path
    :param text: text to input
    """
    with open(dataset_file_path, "a") as f:
        f.write(text)


def parse_filepath(modell_dir):
    file_name = 'grid_search_{0}_{1}.txt'.format(arguments.language, arguments.classifier)
    return os.path.join(modell_dir, file_name)


def get_selected_features(pipe, features):
    """
    Get list of best selected features
    :param pipe: pipeline
    :param features: list of all features
    :return: selcted features
    """
    selected_features = []
    for name, transformer in pipe.steps:
        if name.startswith('feature_selector'):
            X_index = np.arange(len(features)).reshape(1, -1)
            indexes = transformer.transform(X_index).tolist()
            selected_features = features[indexes].tolist()
            print("Selected features: ", selected_features)
    return selected_features


def report_best_results(gs_reports_dir, selected_features, cv_results_, n_best_results=3):
    """
    Write best grid search parameters to file
    :param gs_reports_dir: destination directory path
    :param selected_features: names of features
    :param cv_results_: cross validation results
    :param n_best_results: number of first best results, that should be saved
    """
    logger.info("Saving grid search results:")
    report_filepath = parse_filepath(gs_reports_dir)
    append_to_file(report_filepath, str.format("{0}\n Grid Search Parameter Tuning:\n{0}\n", '=' * 40))

    parameters = str.format("Data set Path: {0} \n"
                            "Data set Language: {1} \n"
                            "Classifier: {2} \n"
                            "Cross-validation: {3} \n"
                            "Scaler: {4} \n"
                            "Normalize: {5}\n"
                            "Feature selector: {6}\n", arguments.train_path, arguments.language, arguments.classifier,
                            arguments.cross_val, arguments.scaler_name, arguments.normalize, arguments.feature_selector)
    append_to_file(report_filepath, parameters)

    if arguments.feature_selector.lower() == "k_best":
        append_to_file(report_filepath, str.format("kBest__n_features: {0}\n", arguments.kBest__n_features))
    elif arguments.feature_selector.lower() == "relieff":
        append_to_file(report_filepath, str.format("relieff__n_features_to_keep: {0}\n"
                                                   "relieff__n_neighbors: {1}\n", arguments.relieff__n_features_to_keep,
                                                   arguments.relieff__n_neighbors))
    elif arguments.feature_selector.lower() == "variance_threshold":
        append_to_file(report_filepath, str.format("variance_thr: ", arguments.variance_thr))

    means = cv_results_['mean_test_score']
    stds = cv_results_['std_test_score']
    params = cv_results_['params']
    results = zip(means, stds, params)
    results = sorted(results, key=itemgetter(0), reverse=True)[:n_best_results]
    append_to_file(report_filepath, str.format("{0}\n Best results:\n{0}\n", '-' * 40))

    for counter, (mean, std, params) in enumerate(results):
        model_rank = "Model rank: {0} \n".format(counter + 1)
        mean_valid_score = "Mean validation score: {0:.3f} (std: {1:.3f}) \n".format(mean, std)
        param_string = "Parameters: {0} \n\n".format(params)
        sel_fetures = "Selected features: {0} \n\n".format(selected_features)
        append_to_file(report_filepath, model_rank)
        append_to_file(report_filepath, mean_valid_score)
        append_to_file(report_filepath, param_string)
        append_to_file(report_filepath, sel_fetures)


def create_gs_reports_dir(project_root_dir):
    """
    create directory to save grid search resulte
    :param project_root_dir:
    :return: created directory name
    """
    root_dir = os.path.join(project_root_dir, "Grid-Search")
    language_dir = os.path.join(root_dir, arguments.language)
    report_dir = os.path.join(language_dir, arguments.classifier)
    create_dir(root_dir)
    create_dir(language_dir)
    create_dir(report_dir)
    return report_dir


if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    # 1) Create directories to save and log grid search results
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    log_dir = os.path.join(project_root_dir, "Log")
    create_dir(project_root_dir)
    create_dir(log_dir)
    log_file = os.path.join(log_dir, 'grid_search__{0}.log'.format(strftime("%Y-%m-%d__%H-%M-%S", gmtime())))
    init_logging(log_dir, log_file)
    logger.info(str.format("Parameter tuning started ....."))
    check_input_params()
    gs_reports_dir = create_gs_reports_dir(project_root_dir)
    selected_features = []
    param_grid = {}

    # 2) Load feature collection
    X_train, y_train, features, _ = load_feature_collection(arguments.train_path)

    # 3) Select classifier and its parameters
    logger.info("Using classifier: {0}".format(arguments.classifier))
    classifier, classif_param_grid = get_classifier(arguments.classifier)

    # 4) Create Pipeline
    pipe = Pipeline(steps=[('classifier', classifier)])

    # 5) Scale data set (if required)
    if arguments.scaler_name is not None:
        X_train = scale(X_train, arguments.scaler_name)

    # 6) Normalize Data set (if required)
    if arguments.normalize:
        X_train = normalize(X_train)

    # 7) Select best features
    if arguments.feature_selector.lower() == "relieff":
        X_train, y_train, selected_features = get_feature_subset(X_train, y_train, features, arguments.relieff__n_neighbors,
                                                     arguments.relieff__n_features_to_keep)
        param_grid = classif_param_grid
    else:
        feature_selector, feature_selector_param_grid = get_selector(arguments.feature_selector, arguments.kBest__n_features,
                                                                     arguments.variance_thr)
        param_grid = dict(feature_selector_param_grid, **classif_param_grid)

    # 8) Build pipeline
    pipeline = build_pipeline()

    # 9) Start grid search
    logger.info("Start grid search")
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1_micro',  # f1_micro
                                   cv=arguments.cross_val, iid=True, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    if arguments.feature_selector.lower() != "relieff":
        selected_features = get_selected_features(grid_search.best_estimator_, features)

    # 10) Save best results
    report_best_results(gs_reports_dir, selected_features, grid_search.cv_results_)
    logger.info("Parameter tuning finished")

    # 11) Log running time
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))