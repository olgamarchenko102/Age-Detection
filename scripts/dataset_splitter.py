import os
import sys
import glob
import time
import json
import codecs
import logging
import argparse
import collections
from glob import glob
from time import gmtime, strftime

from sklearn.model_selection import train_test_split

logger = logging.getLogger()


def config_arg_parser():
    """
    Set Parameters to argument parser
    :return: parse arguments
    """
    parser = argparse.ArgumentParser(description='Splitting data set on test and train.')
    parser.add_argument('-dir_path', required=True, help="root directory, which contains "
                                                         "subdirectories with genre data sets")
    parser.add_argument('-language', required=True, help="dataset language")
    parser.add_argument('-random_state', required=True, type=int, help="")
    return parser.parse_args()

def config(file, log_level=logging.INFO):
    logging.basicConfig(level=log_level, format='%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        filename=file, filemode='w')
    formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler_console = logging.StreamHandler(sys.stdout)  # sys.stdout -> no console output
    handler_console.setFormatter(formatter)
    handler_console.setLevel(log_level)
    logging.getLogger('').addHandler(handler_console)


def init_logging(input_dir, file_name):
    create_dir(input_dir)
    config(file_name, log_level=logging.DEBUG)


def create_dir(dir_path):
    """
    Create directory
    :param dir_path: directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_genre(dir_name):
    """
    extract genre from directory name

    :param dir_name: directory name
    :return: genre
    """
    if "blogs" in dir_name:
        return "blogs"
    elif "twitter" in dir_name:
        return "twitter"
    if "socialmedia" in dir_name:
        return "socialmedia"


def get_author_id(path):
    """
    Parse input document full path
    :param path: document full path
    :return: document id
    """
    elements = path.split("\\")
    id = elements[len(elements) - 1]
    return id.replace(".xml", "")


def extract_label(row):
    """
    Parse row and get label
    :param row: thruth file row
    :return: label
    """
    doc_id, gender, age = row.split(':::')
    age = age.replace("\n","")
    if age == '18-24':
        age_label = 1
    elif age == '25-34':
        age_label = 2
    elif age == '35-49':
        age_label = 3
    elif age == '50-64':
        age_label = 4
    elif age.lower() == '65-xx':
        age_label = 5
    else:
        age_label = 0
        logger.info(str.format('ERROR: detected age {0} is not predifined', age))
    return age_label


def get_label(truth_file, author_id):
    """
    Get label
    :param truth_file: truth file

    :param author_id: author id
    :return: label
    """
    lines = open(truth_file, "r")
    for line in lines:
        if author_id in line:
            age_label = extract_label(line)
            return age_label


def create_corpora_dirs(language_corpus_dir):
    """
    Create directories for text and train datasets

    :return: train directory name,  test directory name
    """
    test__dir = os.path.join(language_corpus_dir, "test-corpus")
    train__dir = os.path.join(language_corpus_dir, "train-corpus")

    create_dir(test__dir)
    create_dir(train__dir)
    return train__dir, test__dir


def report_test_train_splitting(y, y_train, y_test):
    """
    Log splitting results
    :param y: list of full dataset age labels
    :param y_train: list of train dataset age labels
    :param y_test: list of test dataset age labels
    """
    y_sorted = collections.Counter(y)
    y_train_sorted = collections.Counter(y_train)
    y_test_sorted = collections.Counter(y_test)

    logging.info(str.format("-" * 50))
    logger.info("{0:<10}{1:>12}{2:>12}{3:>12}".format("Age label", "Train docs", "Test Docs", "Total docs"))
    logging.info(str.format("-" * 50))
    for x in range(1, 5):
        logger.info("{0:>4}{1:>14}{2:13}{3:>13}".format(x, y_train_sorted[x], y_test_sorted[x],  y_sorted[x]))
    logger.info("{0:>4}{1:>14}{2:>13}{3:>13}".format("5",  y_train_sorted[5], y_test_sorted[5], y_sorted[5]))
    logging.info(str.format("-" * 50))
    logger.info("{0:>5}{1:>14}{2:>13}{3:>13}".format("Sum", len(y_train), len(y_test), len(y)))
    logging.info(str.format("{0}\n", "-" * 50))


def report_genre_splitting(y, X):
    logging.info(str.format("-" * 60))
    logger.info("{0:<10}{1:>10}{2:>16}{3:>12}{4:>10}".format("Age label", "Blogs", "Social media", "Twitter", "Total"))
    logging.info(str.format("-" * 60))

    total_blogs_counter = 0
    total_socialmedia_counter = 0
    total_twitter_counter = 0

    for label in range(1, 6):
        blogs_counter = 0
        socialmedia_counter = 0
        twitter_counter = 0
        indexes = [i for i, e in enumerate(y) if e == label]
        for index in indexes:
            x = X[index]
            genre = x[1]
            if genre is "blogs":
                blogs_counter += 1
            elif genre is "socialmedia":
                socialmedia_counter += 1
            elif genre is "twitter":
                twitter_counter += 1

        logger.info("{0:>4}{1:>14}{2:>13}{3:>13}{4:>13}".format(
            label, blogs_counter, socialmedia_counter, twitter_counter,
            blogs_counter+socialmedia_counter+twitter_counter))

        total_blogs_counter += blogs_counter
        total_socialmedia_counter += socialmedia_counter
        total_twitter_counter += twitter_counter

    logging.info(str.format("-" * 60))
    logger.info("{0:>7}{1:>11}{2:>13}{3:>13}{4:>13}".format(
        "Total", total_blogs_counter, total_socialmedia_counter, total_twitter_counter,
        total_blogs_counter + total_socialmedia_counter + total_twitter_counter))
    logging.info(str.format("{0}\n", "-" * 60))


def merge_datasets(language_corpus_dir):
    """
    1) Go throught each genre directory in root directory,
    extract from each document its author_id, age label, genre
    and build X and y matrices like
    create matrix X = [ [author_1, genre],
                          [author_2, genre],
                          [author_n, genre] ]
    and label matrix y = [label_1, label_x, label_k]
    2) Generate dataset JSON-file

    :return: matrix X, matrix y
    """
    dataset_dict = {}
    X = []
    y = []
    dataset_dirs = []
    genre_dirs = glob(str.format("{0}\*", arguments.dir_path))
    for dir_name in genre_dirs:
        dataset_dirs.append(dir_name)
        logging.info(str.format("Preprocessing {0}", dir_name))
        genre = get_genre(dir_name)
        xml_files = glob(str.format("{0}\*.xml", dir_name))
        truth_file = glob(str.format("{0}\*.txt", dir_name))
        for xml in xml_files:
            author_id = get_author_id(xml)
            age_label = get_label(truth_file[0], author_id)
            X.append([author_id, genre, dir_name])
            y.append(age_label)

            dataset_dict[author_id] = {}
            dataset_dict[author_id]['genre'] = genre
            dataset_dict[author_id]['label'] = age_label
            dataset_dict[author_id]['dir_path'] = dir_name

    # Generate dataset json file
    with open(os.path.join(language_corpus_dir, str.format("full_dataset-{0}.json", arguments.language)), 'w') as outfile:
        json.dump(dataset_dict, outfile)
    return X, y, dataset_dict


def extract_text(source_file):
    """
    Extract text from file
    :param source_file: text file
    :return: extracted text
    """
    content = source_file.readlines()
    if len(content) > 1:
        text = ' '.join(content)
    else:
        text = content[0]
    return text


def distribute_dataset(X, target_dir):
    """
    Distribute full dataset txt-files
    to test and train folders

    :param X: Dataset
    :param target_dir: target folder
    """
    for row in X:
        author_id = row[0]
        file_root = row[2]
        file_name = str.format("{0}.xml", author_id)
        source_file = open(os.path.join(file_root, file_name), "r", encoding='utf-8')
        content = extract_text(source_file)
        target_path = os.path.join(target_dir, file_name)
        # target_file = open(target_path, "w")
        # target_file.write(str(content.encode("utf-8")))
        target_file = codecs.open(target_path, "a", "utf-8")
        target_file.write(str(content))
        target_file.close()


def compile_truth_file(X, y, target_dir):
    """
    Compile new truth file for test or train dataset
    :param X: Dataset
    :param y: labels
    :param target_dir: target dataset directory
    """
    truth_row_list = []
    for index, row in enumerate(X):
        author_id = row[0]
        age_label = y[index]
        truth_row_list.append(str.format("{0}:::{1}", author_id, age_label))
    truth_path = os.path.join(target_dir, "truth.txt")
    truth_file = open(truth_path, "w")
    for element in truth_row_list:
        truth_file.write(str.format("{0}\n", str(element)))
    truth_file.close()



def create_doc_corpus_dir(project_root_dir):
    root_dir = os.path.join(project_root_dir, "Document-Corpus")
    language_dir = os.path.join(root_dir, arguments.language)
    create_dir(root_dir)
    create_dir(language_dir)
    return language_dir

if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    log_dir = os.path.join(project_root_dir, "Log")
    create_dir(project_root_dir)
    create_dir(log_dir)
    log_file = os.path.join(log_dir, 'dataset_splitting__{0}.log'.format(strftime("%Y-%m-%d__%H-%M-%S", gmtime())))
    init_logging(log_dir, log_file)
    logger.info(str.format("Started splitting data set on train and test......"))

    # create directory to save preprocessed files
    language_corpus_dir = create_doc_corpus_dir(project_root_dir)

    # Create folders for train and test data sets
    train_dir, test_dir = create_corpora_dirs(language_corpus_dir)

    # Merge genre data sets
    X, y, dataset_dict = merge_datasets(language_corpus_dir)

    # Split data sets on train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=arguments.random_state)

    # Report splitting results by age labels and genres
    logging.info(str.format("Data distribution by genre in the full data set:"))
    report_genre_splitting(y, X)
    logging.info(str.format("Data distribution on train and test data sets:"))
    report_test_train_splitting(y, y_train, y_test)
    logging.info(str.format("Data distribution by genre in the train data set:"))
    report_genre_splitting(y_train, X_train)
    logging.info(str.format("Data distribution by genre in the test data set:"))
    report_genre_splitting(y_test, X_test)

    # Distribute full dataset txt-files to test and train folders
    distribute_dataset(X_train, train_dir)
    distribute_dataset(X_test, test_dir)

    # Generate truth files for train and test datasets
    compile_truth_file(X_train, y_train, train_dir)
    compile_truth_file(X_test, y_test, test_dir)

    # Log result files and directories
    logging.info(str.format("Training Corpus Directory :{0}", train_dir))
    logging.info(str.format("Test Corpus Directory :{0}", test_dir))
    logger.info(str.format("Log file path: {0}", log_file))
    logger.info("Splitting finished ")

    # Log running time
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))