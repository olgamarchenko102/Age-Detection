
#-dir_path
#"C:\Users\olga-\Desktop\Age-Detection\Feature-Collection\english\train\best_features__svm"
#--------------------------------------------------------------------------------------------------------
import os
import sys
import time
import pandas as pd
import argparse
import logging
from time import gmtime, strftime
import glob
logger = logging.getLogger()


def config_arg_parser():
    """
    Set parameters of argument parser
    :return: parser arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-dir_path', required=True, help="List of Dataframe file paths, that should be merged")
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


def create_dir(dir_path):
    """
    Create directory
    :param dir_path: directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def merge(df_list):
    """
    Merge list of feature collections on author_id key
    :return: path to the merged dataframe
    """

    df_final = pd.read_csv(df_list[0])
    for ind, df in enumerate(df_list):
        if ind >= 1:
            temp_df = pd.read_csv(df_list[ind])
            temp_df = temp_df.drop(['lbl'], axis=1)
            df_final = pd.merge(df_final, temp_df, on=['author_id'])
    final_path = os.path.join(os.path.expanduser("~/Desktop/Age-Detection"), "merged-feature-collection.csv")
    df_final.to_csv(final_path, sep=',', index=False)
    return final_path


if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    log_dir = os.path.join(project_root_dir, "Log")
    create_dir(project_root_dir)
    create_dir(log_dir)
    log_file = os.path.join(log_dir, 'merging_dataframes__{0}.log'.format(strftime("%Y-%m-%d__%H-%M-%S", gmtime())))
    init_logging(log_dir, log_file)
    logger.info(str.format("Started merging dataframes......"))

    csv_files = glob.glob(str.format("{0}\*.csv", arguments.dir_path))
    files = []
    for file in csv_files:
        files.append(os.path.join(arguments.dir_path, file))


    # Start merging
    final_path = merge(files)
    logger.info(str.format("Merged feature collection was saved to: {0}", final_path))
    # Log result files and directories
    logger.info(str.format("Log file path: {0}", log_file))
    logger.info("Datasets have been sucessfully merged!")
    # Log running time
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))



