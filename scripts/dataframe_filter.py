#-feature_collection_path
#"C:\Users\olga-\Desktop\Age-Detection\Feature-Collection\english\train\best_features__svm_linear\12.csv"
#-feature_list
#"abandon,abile"
#-------------------------------------------------------
import time
import os
import sys
import logging
import argparse
import pandas as pd


def config_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-feature_collection_path', required=True, help="")
    parser.add_argument('-feature_list', required=True, help='')
    return parser.parse_args()

logger = logging.getLogger()

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

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_features(csv_path, feature_names, result_path):
    feature_collection = pd.read_csv(csv_path, delimiter=',')

    feature_list = ["author_id", "lbl"] + feature_names
    feature_collection = feature_collection[feature_list]


    feature_collection.to_csv(result_path, sep=',', index=False, encoding='utf-8')
    logger.info(str.format("Extracted features were saved to: {0}", result_path))



if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    create_dir(project_root_dir)

    result_path = os.path.join(project_root_dir, "filtered_features.csv")

    #feature_string = arguments.feature_list.replace(',', ' ')

    feature_list= [x for x in arguments.feature_list.split(',')]
    #feature_list = word_tokenize(feature_string)
    print(feature_list)
    load_features(arguments.feature_collection_path, feature_list, result_path)

    print("Result was saved to", result_path)





