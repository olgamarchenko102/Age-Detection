import os
import sys
import numpy as np
import pandas as pd
import json
import logging
import argparse
import warnings
import time
from time import gmtime, strftime
from operator import itemgetter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
warnings.filterwarnings('ignore')
from ReliefF import ReliefF as reliefF
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger()



def config_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-train_path', required=True, help="")
    parser.add_argument('-language', required=True, help='')
    parser.add_argument('-classifier', required=True, help='')
    parser.add_argument('-cross_val', required=True, type=int, help='n-fold crossvalidation', default=4)
    parser.add_argument('-scaler_name', required=False, help='')
    parser.add_argument('-normalize', required=True, help='', default=True)
    parser.add_argument('-feature_selector', required=False, help='', default='relieff')
    parser.add_argument('-kBest__n_features', required=False, type=int, help='', default=3)
    parser.add_argument('-variance_thr', type=float, required=False, help='value for variance threshold', default=0.0) #0.8*(1- 0.8)
    parser.add_argument('-relieff__n_features_to_keep', required=False, type=int, help='', default=1)
    parser.add_argument('-relieff__n_neighbors', type=int, required=False, help='', default=3)
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




# Region "Loader" =================================
def load_features(csv_path, feature_names):
    feature_names[:0] = ["lbl"]
    feature_names[:0] = ["author_id"]
    feature_collection = pd.read_csv(csv_path, delimiter=',')
    data = np.array(feature_collection[feature_names[2:]][0:], dtype=np.float)
    # TODO check if data = 0
    categories = np.array(feature_collection["lbl"][0:], dtype=np.int)
    return data, categories


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


def load_feature_collection(csv_path):
    feature_names = get_column_names(csv_path)
    # print(feature_names)
    # print(len(feature_names))
    feature_collection = pd.read_csv(csv_path, delimiter=',')
    data = np.array(feature_collection[feature_names[2:]][0:])  # , dtype=np.float
    categories = np.array(feature_collection["lbl"][0:], dtype=np.int)
    author_ids = np.array(feature_collection["author_id"][0:])
    header_row = np.array(feature_names)
    features = header_row[2:]
    return data, categories, features, author_ids


# ==================================================


# Region "Scaler" =================================
def get_scaler(scaler_name):
    print("Applying ", scaler_name)
    if scaler_name == "MinMaxScaler":
        return MinMaxScaler(feature_range=(0, 1))
    elif scaler_name == "StandardScaler":
        return StandardScaler(with_mean=True, with_std=True)


def scale_train_test(X_train, X_test, scaler_name):
    scaler = get_scaler(scaler_name)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def scale(X_train, scaler_name):
    scaler = get_scaler(scaler_name)
    return scaler.fit_transform(X_train)

# ==================================================

def knn_param_grid():
    return {
        "classifier__n_neighbors": [30,35],
        "classifier__leaf_size": [5], #30
        "classifier__weights": ["uniform"]
        #
        # "classifier__n_neighbors": [9, 11, 15, 17, 21, 23, 25, 30, 35],
        # "classifier__leaf_size": [5, 10, 15, 20, 25],  # 30
        # "classifier__weights": ["distance", "uniform"]
    }


def knn_build(n_neighbors=5, leaf_size=30, weights='uniform'):
    return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, weights=weights)


def svm_build(kernel, random_state=0, C=1.0, gamma='auto'):
    return SVC(kernel=kernel, class_weight='balanced', random_state=random_state, C=C, gamma=gamma)


def svm_param_grid_rbf():
    # c_params = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    # gamma_params = [ 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

    # c_params = [1e0, 1e1 ]
    # gamma_params = [1e-2, 1e-1, 1e0, ]
    c_params = [1e1]
    gamma_params = [ 1e-1]
    return {'classifier__C': c_params,
            'classifier__gamma': gamma_params}


def svm_param_grid_linear():
    c_params = [ 1e1] #  c_params = [ 1e-2, 1e-1, 1e0, 1e2, 1e3, 5e3] 1e-1, 1e0,
    return {'classifier__C': c_params}


def random_forest_param_grid():
    return {"classifier__n_estimators": [10, 15, 20, 50],
            "classifier__max_depth": [25] #, 50, 100
            # "classifier__max_features": [1, 0.7, 0.5, 0.4, 0.2],
            # "classifier__min_samples_split": [2, 5, 7, 9],
            # "classifier__min_samples_leaf": [1, 5, 7]
            }


def random_forest_build(random_state=0, n_jobs=1, min_samples_split=2, n_estimators=100, max_depth=None, max_features='auto',
          min_samples_leaf=1, bootstrap=True, criterion='gini'):
    return RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=n_jobs,
                                  min_samples_split=min_samples_split, n_estimators=n_estimators, max_depth=max_depth,
                                  max_features=max_features, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                                  criterion=criterion)


def logistic_regression_param_grid():
    return {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": ["balanced", None]
            }


def logistic_regression_build(penalty='l2', multi_class='multinomial', C=10, class_weight="balanced", solver="lbfgs",max_iter=20):
    return LogisticRegression(penalty=penalty, multi_class=multi_class,
                              C=C, class_weight=class_weight, solver=solver, max_iter=max_iter)


def get_classifier(clf_name):
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
    relief = reliefF(n_neighbors=n_neighbors, n_features_to_keep=relieff__n_features_to_keep)
    X_subset = relief.fit_transform(X,y)
    feature_indexes = relief.top_features[:relieff__n_features_to_keep]
    selected_features = features[feature_indexes]
    return X_subset, y, selected_features





def k_best_build():
    return SelectKBest()


def k_best_param_grid(kBest_n_features):
    return {"feature_selector__score_func": [f_classif],
            "feature_selector__k": [kBest_n_features]
            }


def variance_threshold_build():
    return VarianceThreshold()

def variance_threshold_param_grid(variance_thr):
    return {"feature_selector__threshold": [variance_thr]}



def get_selector(selector_name, kBest_n_features, variance_thr):
    if selector_name.lower() == "k_best":
        return k_best_build(), k_best_param_grid(kBest_n_features=kBest_n_features)
    elif selector_name.lower() == "variance_threshold":
        return variance_threshold_build(), variance_threshold_param_grid(variance_thr=variance_thr)
    else:
        raise ValueError("Unknown selector")


def build_pipeline():
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
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def check_input_params():
    if arguments.language.lower() not in ['english', 'spanish']:
        logger.info('ERROR: Wrong language input. Try English or Spanish')
        sys.exit(1)


def append_to_file(dataset_file_path, text):
    with open(dataset_file_path, "a") as f:
        f.write(text)


def parse_filepath(modell_dir):
    file_name = 'grid_search_{0}_{1}.txt'.format(arguments.language, arguments.classifier)
    return os.path.join(modell_dir, file_name)



def get_selected_features(pipe, features):
    selected_features = []
    for name, transformer in pipe.steps:
        if name.startswith('feature_selector'):
            X_index = np.arange(len(features)).reshape(1, -1)
            indexes = transformer.transform(X_index).tolist()
            selected_features = features[indexes].tolist()
            print("Selected features: ", selected_features)
    return selected_features


# def save_best_params(gs_reports_dir, params, features):
#
#     settings = {
#                 "classifier": arguments.classifier,
#                 "normalize": bool(arguments.normalize),
#                 "features": features
#                 }
#     if arguments.scaler_name is None:
#         settings["scaler_name"] = ""
#     else:
#         settings["scaler_name"] = arguments.scaler_name
#
#     if arguments.classifier == "knn":
#         settings["n_neighbors"] = int(params["classifier__n_neighbors"])
#         settings["leaf_size"] = int(params["classifier__leaf_size"])
#         settings["weights"] = params["classifier__weights"]
#     elif arguments.classifier == "svm":
#         settings["C"] = params["classifier__C"]
#         settings["gamma"] = params["classifier__gamma"]
#         settings["kernel"] = "rbf"
#     elif arguments.classifier == "svm-linear":
#         settings["C"] = params["classifier__C"]
#         settings["kernel"] = "linear"
#     elif arguments.classifier == "random_forest":
#         settings["bootstrap"] = bool(params["classifier__bootstrap"])
#         settings["criterion"] = params["classifier__criterion"]
#         settings["max_depth"] = int(params["classifier__max_depth"])
#         settings["max_features"] = float(params["classifier__max_features"])
#         settings["min_samples_leaf"] = int(params["classifier__min_samples_leaf"])
#         settings["min_samples_split"] = int(params["classifier__min_samples_split"])
#         settings["n_estimators"] = int(params["classifier__n_estimators"])
#     elif arguments.classifier == "logistic_regression":
#         settings["C"] = float(params["classifier__C"])
#         if params["classifier__class_weight"] is not None:
#             settings["class_weight"] = params["classifier__class_weight"]
#
#     else:
#         raise ValueError("Unknown classifier")
#
#
#     settings_file = os.path.join(gs_reports_dir, "{0}_{1}.json".format(arguments.language, arguments.classifier))
#     with open(settings_file, 'w') as outfile:
#         json.dump(settings, outfile)
#
#     logger.info(str.format("Tuned parameters were saved to: {0}", gs_reports_dir))


# write to report file top 3 best results
def report_best_results(gs_reports_dir, selected_features, cv_results_, n_best_results=3):
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

        # #output_features = []
        # if counter == 0:  # first best result
        #     if isinstance(selected_features, list):
        #         output_features = selected_features
        #     else:
        #         output_features = selected_features.tolist()
        #     save_best_params(gs_reports_dir, params, output_features)

def create_gs_reports_dir(project_root_dir):
    root_dir = os.path.join(project_root_dir, "Grid-Search")
    language_dir = os.path.join(root_dir, arguments.language)
    klassifikator = os.path.join(language_dir, arguments.classifier)
    create_dir(root_dir)
    create_dir(language_dir)
    create_dir(klassifikator)
    return klassifikator


if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
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

    X_train, y_train, features, _ = load_feature_collection(arguments.train_path)

    # 2) Select classifier and its parameters
    logger.info("Using classifier: {0}".format(arguments.classifier))
    classifier, classif_param_grid = get_classifier(arguments.classifier)
    # 3) Create Pipeline
    pipe = Pipeline(steps=[('classifier', classifier)])
    # Scale data set (if required)
    if arguments.scaler_name is not None:
        X_train = scale(X_train, arguments.scaler_name)
    # Normalize Data set (if required)
    if arguments.normalize:
        X_train = normalize(X_train)

        # 5) Scale data set if requiered and get data subset
    if arguments.feature_selector.lower() == "relieff":
        X_train, y_train, selected_features = get_feature_subset(X_train, y_train, features, arguments.relieff__n_neighbors,
                                                     arguments.relieff__n_features_to_keep)
        param_grid = classif_param_grid
    else:
        feature_selector, feature_selector_param_grid = get_selector(arguments.feature_selector, arguments.kBest__n_features,
                                                                     arguments.variance_thr)
        param_grid = dict(feature_selector_param_grid, **classif_param_grid)

    # 6) Build pipeline
    pipeline = build_pipeline()


    # 7) Start grid search
    logger.info("Start grid search")
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1_micro',  # f1_micro
                                   cv=arguments.cross_val, iid=True, n_jobs=1, verbose=2)

    grid_search.fit(X_train, y_train)


    if arguments.feature_selector.lower() != "relieff":
        selected_features = get_selected_features(grid_search.best_estimator_, features)


    # 8) Save best results to json-File
    report_best_results(gs_reports_dir, selected_features, grid_search.cv_results_)
    logger.info("Parameter tuning finished")

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))




