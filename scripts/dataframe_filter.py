import os
import sys
import time
import logging
import argparse
import pandas as pd

logger = logging.getLogger()


def config_arg_parser():
    """
    Set Parameters to argument parser
    :return: parse arguments
    """
    parser = argparse.ArgumentParser(description='Dataframe Filter')
    parser.add_argument('-feature_collection_path', required=True, help="Feature collection csv-file path")
    parser.add_argument('-feature_list', required=True, help='List of the features to select')
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


def create_dir(directory):
    """
    Create directory
    :param dir_path: directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def filtered_feature_collection(csv_path, feature_names, result_path):
    """
    Save filtered feature collection to the new file
    :param csv_path: source file path
    :param feature_names: list of feature to select
    :param result_path: result file path
    """
    feature_collection = pd.read_csv(csv_path, delimiter=',')
    feature_list = ["author_id", "lbl"] + feature_names
    feature_collection = feature_collection[feature_list]
    feature_collection.to_csv(result_path, sep=',', index=False, encoding='utf-8')
    logger.info(str.format("Extracted features were saved to: {0}", result_path))


if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    # 1) Create directory to save filtered feature collection
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    create_dir(project_root_dir)
    result_path = os.path.join(project_root_dir, "filtered_features.csv")
    feature_list = [x for x in arguments.feature_list.split(',')]
    # 2) Filter feature collection by input list of features
    filtered_feature_collection(arguments.feature_collection_path, feature_list, result_path)
    print("Result was saved to", result_path)





