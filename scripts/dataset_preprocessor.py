import os
import sys
import glob
import time
import codecs
import logging
import argparse
import xml.etree.ElementTree as elemTree
from time import gmtime, strftime

logger = logging.getLogger()


def config_arg_parser():
    """
    Set Parameters to argument parser
    :return: parser arguments
    """
    parser = argparse.ArgumentParser(description='Dataset Preprocessor')
    parser.add_argument('-dir_path', required=True, help="Raw data directory path")
    parser.add_argument('-language', required=True, help="Dataset language")
    return parser.parse_args()


def get_xml_root(full_path):
    """
    Get xml root
    :param full_path: directory path
    :return: xml root
    """
    tree = elemTree.parse(full_path)
    return tree.getroot()


def get_doc_id(path):
    """
    Parse input document full path
    :param path: document full path
    :return: document id
    """
    elements = path.split("\\")
    return elements[len(elements) - 1]


def append_to_file(dataset_file_path, text):
    """
    Write input text to input file
    :param dataset_file_path:  destination file path
    :param text: text
    """
    file = codecs.open(dataset_file_path, "a", "utf-8")
    file.write(text)
    file.close()


def extract_messages(root):
    """
    Extract HTML-Content from Document
    :param root: xml-root element
    :return: list of texts
    """
    messages = []
    for node in root.iter('document'):
        if node.text is not None:
            messages.append(node.text)
    return messages


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


def create_document_corpus_dir(project_root_dir):
    """
    Create document corpus directory
    :param project_root_dir: root project directory
    """
    corpus_root_dir = os.path.join(project_root_dir, "Preprocessed-Corpus")
    language_dir = os.path.join(corpus_root_dir, arguments.language.lower())
    create_dir(corpus_root_dir)
    create_dir(language_dir)
    return language_dir


class DataSetPreprocessor(BaseException, ):
    """
    The class DataSetCompiler Extracts textdata
    from input files  and builds a Dataset
    """
    def __init__(self):
        """
        Initialising function
        """
        self.sub_dir = ''
        self.messages_counter = {}

    def create_dataset_dir(self, document_corpus_dir):
        """
        Create  root directory to save the preprocessed dataset.
        Create subdirectory for resonsive genre & language dataset
        """
        dir_name = os.path.basename(os.path.normpath(arguments.dir_path))

        self.sub_dir = os.path.join(document_corpus_dir, str.format("{0}", dir_name))
        #create_dir(data_set_dir)
        create_dir(self.sub_dir)

    def create_ds_document(self, messages, author_id):
        """
        Create a dataset document, which is named by
        user_id and incudes all messaged of one specific author
        :param messages: list of all author messages
        :param author_id: autor_id
        """
        path = os.path.join(self.sub_dir, author_id)
        doc_text = ' '.join(messages)
        append_to_file(path, doc_text)

    def log_reports(self, total_docs):
        """
        Log documents' analyse statistics
        :param total_docs: number of documents (int)

        """
        logger.info(str.format("Data set path: {0}", arguments.dir_path))
        logger.info(str.format("Total number of .xml-files: {0}", total_docs))
        logger.info(
            str.format("{2}\n{1}Author_id{0}Number_of_messages\n{3}{2}", " " * 19, " " * 30, "=" * 63, " " * 21))

        sorted_messages_counter = sorted(self.messages_counter.items(), key=lambda x: x[1], reverse=True)
        for doc_key, number in sorted_messages_counter:
            logger.info(str.format("{0}{1}{2}", doc_key.replace('.xml', ''), " " * 9, number))
        logger.info("=" * 63)

    def compile(self, document_corpus_dir):
        """
        Filter HTML-text from xml-files and
        join all messaged from one author to one txt-file.
        Count number of messages in each document.
        """
        self.create_dataset_dir(document_corpus_dir)
        xml_files = glob.glob(str.format("{0}\*.xml", arguments.dir_path))

        for path in xml_files:
            doc_id = get_doc_id(path)
            root = get_xml_root(path)
            messages = extract_messages(root)
            self.create_ds_document(messages, doc_id)
            self.messages_counter[doc_id] = len(messages)
        truth_file = "truth.txt"
        source_file = open(os.path.join(arguments.dir_path, truth_file), "r", encoding='utf-8')
        lines = source_file.readlines()
        target_file = open(os.path.join(self.sub_dir, truth_file), "w")
        for line in lines:
            target_file.write(line)
        target_file.close()
        self.log_reports(len(xml_files))


if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()

    # 1) Create directories to save and log the preprocessed files
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    log_dir = os.path.join(project_root_dir, "Log")
    create_dir(project_root_dir)
    create_dir(log_dir)
    log_file = os.path.join(log_dir, 'dataset_compiling__{0}.log'.format(strftime("%Y-%m-%d__%H-%M-%S", gmtime())))
    init_logging(log_dir, log_file)
    logger.info(str.format("Started compiling data set......"))
    document_corpus_dir = create_document_corpus_dir(project_root_dir)

    # 2) Star Preprocessing
    ds = DataSetPreprocessor()
    ds.compile(document_corpus_dir)

    # 3) Log result files and directories
    logger.info(str.format("Preprocessed documents have been saved to : {0}", document_corpus_dir))
    logger.info(str.format("Log file path: {0}", log_file))
    logger.info("Compiling finished")

    # 4) Log running time
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))