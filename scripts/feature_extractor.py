import time
import os
import re
import sys
import math
import glob
import enchant
import logging
import argparse
import textstat
import unicodedata
import numpy as np
import pandas as pd
import treetaggerwrapper
from random import randint
from nltk.corpus import stopwords
from time import gmtime, strftime
# from spacy.lang.en import English
# from spacy.lang.es import Spanish
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
# pip install http://pypi.python.org/packages/source/h/htmllaundry/htmllaundry-2.0.tar.gz
from htmllaundry import strip_markup
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()

# https://github.com/Alir3z4/stop-words/blob/bd8cc1434faeb3449735ed570a4a392ab5d35291/english.txt
STOP_WORDS_EN = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
                 "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                 "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
                 "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
                 "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
                 "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
                 "it", "it's", "its", "itself", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
                 "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
                 "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
                 "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
                 "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
                 "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
                 "we're",
                 "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
                 "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
                 "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
# https://github.com/Alir3z4/stop-words/blob/bd8cc1434faeb3449735ed570a4a392ab5d35291/spanish.txt
STOP_WORDS_ES = ["a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra", "cual", "cuando",
                 "de", "del", "desde", "donde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era",
                 "erais", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba",
                 "estabais", "estaban", "estabas", "estad", "estada", "estadas", "estado", "estados", "estamos",
                 "estando", "estar", "estaremos", "estará", "estarán", "estarás", "estaré", "estaréis", "estaría",
                 "estaríais", "estaríamos", "estarían", "estarías", "estas", "este", "estemos", "esto", "estos",
                 "estoy", "estuve", "estuviera", "estuvierais", "estuvieran", "estuvieras", "estuvieron", "estuviese",
                 "estuvieseis", "estuviesen", "estuvieses", "estuvimos", "estuviste", "estuvisteis", "estuviéramos",
                 "estuviésemos", "estuvo", "está", "estábamos", "estáis", "están", "estás", "esté", "estéis", "estén",
                 "estés", "fue", "fuera", "fuerais", "fueran", "fueras", "fueron", "fuese", "fueseis", "fuesen",
                 "fueses", "fui", "fuimos", "fuiste", "fuisteis", "fuéramos", "fuésemos", "ha", "habida", "habidas",
                 "habido", "habidos", "habiendo", "habremos", "habrá", "habrán", "habrás", "habré", "habréis", "habría",
                 "habríais", "habríamos", "habrían", "habrías", "habéis", "había", "habíais", "habíamos", "habían",
                 "habías", "han", "has", "hasta", "hay", "haya", "hayamos", "hayan", "hayas", "hayáis", "he", "hemos",
                 "hube", "hubiera", "hubierais", "hubieran", "hubieras", "hubieron", "hubiese", "hubieseis", "hubiesen",
                 "hubieses", "hubimos", "hubiste", "hubisteis", "hubiéramos", "hubiésemos", "hubo", "la", "les", "lo",
                 "los", "me", "mi", "mis", "mucho", "muchos", "muy", "más", "mías", "mío", "míos", "nada", "ni", "no",
                 "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "o", "os", "otra",
                 "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "que", "quien", "quienes", "qué",
                 "se", "sea", "seamos", "sean", "seas", "seremos", "será", "serán", "serás", "seré", "seréis", "sería",
                 "seríais", "seríamos", "serían", "serías", "seáis", "sido", "siendo", "sin", "sobre", "sois", "somos",
                 "son", "soy", "su", "sus", "suya", "suyas", "suyo", "suyos", "sí", "también", "tanto", "te",
                 "tendremos", "tendrá", "tendrán", "tendrás", "tendré", "tendréis", "tendría", "tendríais",
                 "tendríamos",
                 "tendrían", "tendrías", "tened", "tenemos", "tenga", "tengamos", "tengan", "tengas", "tengo",
                 "tengáis",
                 "tenida", "tenidas", "tenido", "tenidos", "teniendo", "tenéis", "tenía", "teníais", "teníamos",
                 "tenían", "tenías", "ti", "tiene", "tienen", "tienes", "todo", "todos", "tu", "tus", "tuve", "tuviera",
                 "tuvierais", "tuvieran", "tuvieras", "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses",
                 "tuvimos", "tuviste", "tuvisteis", "tuviéramos", "tuviésemos", "tuvo", "tuya", "tuyas", "tuyo",
                 "tuyos",
                 "tú", "un", "una", "uno", "unos", "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros",
                 "y", "ya", "yo", "él", "éramos"]
EMOTICON_DICT = [':-)', ':)', ':]', '=)', ':-(', ':(', ':[', '=(', ';-)', ';)', ':-d', ':d', '=d', ':-p',
                  ':p', '=p', ':3', ':-*', ':*', '^_^', '^^', '-_-', '>:o', '>:-o', '>:(', '>:-(', ':-o',
                  ':o', ":'(", ":'-(", ':/', ':-/', '3:)', '3:-)', '(^^^)', '<3', 'o:)',
                  'o:-)', '8-)', '8)', 'b-)', 'b)', '8-|', '8|', 'b-|', 'b|', ':|', ':-|', ':@']
NOUNS_EN = ['NN', 'NNS', 'NP', 'NPS']
VERBS_EN = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VD', 'VDD', 'VDG', 'VDN', 'VDZ', 'VDP', 'VH',
            'VHD', 'VHG', 'VHN', 'VHZ', 'VHP', 'VV', 'VVD', 'VVG', 'VVN', 'VVP', 'VVZ']
ADJECTIVES_EN = ['JJ', 'JJR', 'JJS']
ADVERBS_EN = ['RB', 'RBR', 'RBS', 'WRB']
PREPOSITIONS_EN = ['IN']
PRONOUNS_EN = ['PP', 'PP$', 'WP', 'WP$']
DETERMINERS_EN = ['DT', 'WDT', 'PDT']
PARTICLES_EN = ['RP']
FOREIGN_WORDS_EN = ['FW']
NOUNS_ES = ['NC', 'NMEA', 'NMON', 'NP']
VERBS_ES = ['VCLICger', 'VCLICinf', 'VCLICfin', 'VEadj', 'VEfin', 'VEinf', 'VEger', 'VHadj', 'VHfin', 'VHger', 'VHinf',
            'VLadj', 'VLfin', 'VLger', 'VLinf', 'VMadj', 'VMfin', 'VMger', 'VMinf', 'VSadj', 'VSfin', 'VSger', 'VSinf',
            'VCLIinf', 'VCLIger', 'VCLIfin']
ADJECTIVES_ES = ['ADJ', 'ORD', 'QU']
ADVERBS_ES = ['ADV', 'NEG']
ADPOSITIONS_ES = ['PAL', 'PDEL', 'PREP', 'PREP/DEL']  # prepositions and endpositions
PRONOUNS_ES = ['DM', 'INT', 'PPC', 'PPO', 'PPX', 'SE']
DETERMINERS_ES = ['ART']
FOREIGN_WORDS_ET_AL_ES = ['ITJN', 'ACRNM', 'ALFP', 'ALFS', 'CODE', 'FO', 'PE', 'PNC', 'SYM',
                          'UMMX']  # foreign words, typos, abbreviations


def config_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-doc_dir', required=True, help="")
    parser.add_argument('-language', required=True, help='')
    parser.add_argument('-dataset_type', required=True, help='')
    parser.add_argument('-feature_name', required=True, help='')
    parser.add_argument('-ngram', required=False, type=int, default=1, help='')
    parser.add_argument('-min_df', required=False, type=float, default=1.0, help='')
    parser.add_argument('-max_df', required=False, type=float, default=1.0, help='')
    parser.add_argument('-max_features', required=False, type=int, default=None, help='')
    parser.add_argument('-text_preprocessing', required=True,  default='stemm', help='lemma or stemm')
    return parser.parse_args()


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


def extract_document_label(element):
    doc_id, age_label = element.split(':::')
    return doc_id, age_label


def get_raw_text(root_path, xml_files, author_id):
    html_text = extract_document_text(root_path, xml_files, author_id)
    text = remove_tabs_and_newlines(html_text)
    text = unescape_html(text)
    text = remove_urls(text)
    text = strip_markup(text) #clean all the HTML markups, this function is a part of htmllaundry
    text = remove_html_formatting(text)

    return text


def unescape_html(text):
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&apos;", "'")
    text = text.replace("&amp;", "&")
    return text

def remove_tabs_and_newlines(text):
    """
    Replace tabs (\t) and  newlines (\n) by whitespace charachter

    :param text: input text
    :return: text without tabs and newlines
    """
    return re.sub('\s+', ' ', text)


def remove_urls(text):
    text = re.sub('<[aA] (href|HREF)=.*?</[aA]>;?', ' ', text)
    text = re.sub('<img.*?>;?',' ', text)
    text = re.sub('(http|https|ftp)://?[0-9a-zA-Z\.\/\-\_\?\:\=]*', ' ', text)
    text = re.sub('(http|https|ftp)://?[0-9a-zA-Z\.\/\-\_\?\:\=]*', ' ', text)
    text = re.sub('(^|\s)www\..+?(\s|$)', ' ', text)
    text = re.sub('(^|\s)(http|https|ftp)\:\/\/t\.co\/.+?(\s|$)', ' ', text)
    text = re.sub('(^|\s)(http|https|ftp)\:\/\/.+?(\s|$)', ' ', text)
    text = re.sub('(^|\s)pic.twitter.com/.+?(\s|$)', ' ', text)
    return text


def remove_html_formatting(text):
    # get rid of bbcode formatting and remaining html markups
    text = re.sub('[\[\<]\/?b[\]\>];?', ' ', text)
    text = re.sub('[\[\<]\/?p[\]\>];?', ' ', text)
    text = re.sub('[\[\<]\/?i[\]\>];?', ' ', text)
    text = re.sub('[\[\<]br [\]\>];?', ' ', text)
    text = re.sub('/>', ' ', text)
    text = re.sub('[\<\[]\/?h[1-4][\>\]]\;?', ' ', text)
    text = re.sub('\[\/?img\]', ' ', text)
    text = re.sub('\[\/?url\=?\]?', ' ', text)
    text = re.sub('\[/?nickname\]', ' ', text)
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(cleanr, ' ', text)

    # delete everything else that strip_markup doesn't
    text = re.sub('height=".*?"', ' ', text)
    text = re.sub('width=".*?"', ' ', text)
    text = re.sub('alt=".*?"', ' ', text)
    text = re.sub('title=".*?"', ' ', text)
    text = re.sub('border=".*?"', ' ', text)
    text = re.sub('align=".*?', ' ', text)
    text = re.sub('style=".*?"',' ', text)
    text = re.sub(' otted  border-color:.*?"', ' ', text)
    text = re.sub(' ashed  border-color:.*?"', ' ', text)
    text = re.sub('target="_blank">',' ', text)
    text = re.sub('<a target=" _new"  href="  ]', ' ', text)
    text = re.sub('<a target="_new" rel="nofollow" href=" ]', ' ', text)
    return text


def extract_document_text(root_path, xml_files, doc_id):
    doc_path = os.path.join(root_path, str.format('{0}.xml', doc_id))
    if doc_path in xml_files:
        with open(doc_path, 'r', encoding="utf8") as file:
            html_text = file.read()
            return html_text
    else:
        logger.info('ERROR: doc_path ', doc_path, 'does not exist')



def remove_digits(text):
    """
    Remove number charachters

    :param text: input text
    :return: text without number characters
    """
    reg = re.compile('[0-9]')
    return reg.sub('', text)


def strip_white_spaces(text):
    return text.replace(' ', '')


def strip_punctuation(text):
    text = re.sub(u"[^\w\d'\s]+", ' ', text)
    text = text.replace("'", "")
    text = text.replace("\\", " ")
    reg = re.compile('_')
    text = reg.sub(' ', text)
    text = re.sub(' +', ' ', text)

    return text


def strip_apostroph(text):
    return text.replace("‘", '').replace("’", '').replace("'", '')


def remove_non_alphabetic_chars(raw_text):
    reg = re.compile('[^a-zA-Z]')
    alphabetic_chars_only = reg.sub(' ', raw_text)
    return remove_multiple_spaces(alphabetic_chars_only)


def remove_multiple_spaces(text):
    """
    Replace multiple whitespaces by one
    :param text: input text
    :return: text without multiple whitespaces
    """
    return re.sub(' +', ' ', text)



def remove_special_characters(text):
    return re.sub('[^A-Za-z0-9]+', ' ', text)



def remove_user_mentions(text):
    return re.sub('(^|\s)@(?!\s).+?(?=(\s|$))', '', text)


def strip_multiple_whitespaces(text):
    return re.sub('\s+', ' ', text).strip()


def strip_accent_mark(text):
    # return unidecode.unidecode(text)
    form = unicodedata.normalize('NFKD', text)
    return form.encode('ASCII', 'ignore').decode('ASCII')


def strip_punctuation_but_apostrophs(text):
    text = re.sub(u"[^\w\d'\d’\s]+", ' ', text)
    reg = re.compile('_')
    text = reg.sub(' ', text)
    return text.replace('  ', ' ')


# def strip_user_mentions(text):
#     return re.sub(r'(?:@[\w_]+)', u'', text)


def strip_non_latin(text):
    text = re.sub(r'[\u0627-\u064a]', u'', text)
    return re.sub(r'[\u0600-\u06FF]', u'', text)


def alter_short_word_forms(text):
    text = text.replace("can't", "can not")
    text = text.replace("won't", "will not")
    text = text.replace("n't", " not")
    text = text.replace("'ve", " have")
    text = text.replace("'d", " would")
    text = text.replace("'m", " am")
    text = text.replace("'ll", " will")
    text = text.replace("'s", " has")
    text = text.replace("'re", " are")
    return text


def alter_apostroph_prefix(text):
    text = text.replace("can't", "can")
    text = text.replace("won't", "will")
    text = text.replace("n't", "")
    text = text.replace("'ve", "")
    text = text.replace("'d", "")
    text = text.replace("'m", "")
    text = text.replace("'ll", "")
    text = text.replace("'s", "")
    text = text.replace("'re", "")
    return text

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# def compile_file_path(language):
#     models_dir = os.path.join(os.path.expanduser("~/Desktop"), 'Feature-Collection')
#     create_dir(models_dir)
#     file_name = 'feature-collection-{0}.csv'.format(language)
#     return os.path.join(models_dir, file_name)



def log_dataset_description():
    feature = arguments.feature_name
    logger.info("=" * 70)
    logger.info("DATASET DESRIPTION")
    logger.info(str.format("Dataset Language: {0}", arguments.language))
    logger.info(str.format("Document Corpus Path: {0}", arguments.doc_dir))
    logger.info(str.format("Feature Name: {0}", feature.upper()))
    if feature == "n_grams":
        logger.info(str.format("N: {0}", arguments.ngram))
        logger.info(str.format("Min_df: {0}", arguments.min_df))
        logger.info(str.format("Max_df: {0}", arguments.max_df))
        logger.info(str.format("Max_features: {0}", arguments.max_features))
        if arguments.text_preprocessing.lower() is "lemma":
            logger.info(str.format("Applying Lemmatizing"))
        elif arguments.text_preprocessing.lower() is "stemm":
            logger.info(str.format("Applying Stemming"))
    logger.info("=" * 70)


def get_stop_words(lang):
    if lang is "en":
        return STOP_WORDS_EN
    if lang is "es":
        return STOP_WORDS_ES


def get_sw_list(language):
    """
    Compile list of stopwords by combining
    two lists of stopwords from nltk and
    stop_words libraries
    :param language: dataset language
    :return: compiled stopwords list
    """
    lang = get_lang(language)
    nltk_stop_words = list(stopwords.words(language))
    stop_words = get_stop_words(lang)
    return combine(nltk_stop_words, stop_words)


def get_lang(language):
    """
    Get short cut of input language
    :param language: language
    :return: short cut
    """
    if language == 'english':
        return 'en'
    elif language == 'spanish':
        return 'es'


def combine(list_1, list_2):
    """
    Merge two lists without duplicates
    :param list_1: first list
    :param list_2: secont list
    :return: merged list
    """
    return list(set(list_1) | set(list_2))


def stemm_spanish_words(tokens):
    result_tokens = []
    stemmer = SnowballStemmer('spanish')
    for word in tokens:
        result_tokens.append(stemmer.stem(word))
    return result_tokens


def stemm_english_words(tokens):
    result_tokens = []
    stemmer = PorterStemmer()
    for word in tokens:
        result_tokens.append(stemmer.stem(word))
    return result_tokens


def preprocess_text_for_ngramms(text, language):
    if language == 'english':
        text = alter_short_word_forms(text)
    #text = remove_urls(text)
    text = strip_punctuation(text)
    text = remove_digits(text)
    text = text.replace("'", "")
    return text


def preprocess_text_for_emoticons(text):
    text = remove_urls(text)
    text = strip_accent_mark(text)
    text = text.lower()
    return text



def preprocess_text_for_bow(text):
    #text = remove_tabs_and_newlines(text)
    #text = remove_urls(text)
    text = alter_apostroph_prefix(text)
    text = strip_accent_mark(text)
    text = strip_non_latin(text)
    text = strip_punctuation(text)
    text = remove_digits(text)
    text = remove_multiple_spaces(text)
    text = text.lower()
    return text



def lemmatise_tokens(bag_of_words, nlp):
    lemmas = []
    joined_tokens = " ".join(bag_of_words)
    document = nlp(joined_tokens)
    for word in document:
        lemmas.append(word.lemma_)
    return lemmas


def convert_lang(lang):
    language = 'en'
    if lang.lower() == "spanish":
        language = 'es'
    return language


def check_spelling(tokens, language):
    spell = SpellChecker(convert_lang(language))
    unknown_tokens = list(spell.unknown(tokens))
    return list(spell.unknown(unknown_tokens))


def get_misspelled_tokens(tokens, language):
    misspelled = []
    dictionary = enchant.Dict(get_lang(language))
    for token in tokens:
        if not dictionary.check(token):
            misspelled.append(token)
    return misspelled


def count_ratio(total_tokens, misspelled_tokens):
    ratio = round((misspelled_tokens / total_tokens), 2)
    return ratio


def stemm_tokens(language, bag_of_words):
    stemmed_tokens = []
    if language == 'spanish':
        stemmed_tokens = stemm_spanish_words(bag_of_words)
        # stemmed_tokens = delete_stop_words(language, stemmed_tokens)
    elif language == 'english':
        stemmed_tokens = stemm_english_words(bag_of_words)
        # stemmed_tokens = delete_stop_words(language, stemmed_tokens)
    return stemmed_tokens


def get_bow(text):
    bag_of_words = word_tokenize(text)
    bag_of_words = [element.lower() for element in bag_of_words]
    #bag_of_words[0] = bag_of_words[0][1:]
    bag_of_words = [x for x in bag_of_words if x]
    bag_of_words = [x for x in bag_of_words if not (x.isdigit()
                               or x[0] == '-' and x[1:].isdigit())]
    return bag_of_words


def log_processing_progress(index, corpus_size):
    """
    Logging document preprocessing progress

    :param index: document index
    :param corpus_size: size of corpus (number of documents)
    """
    if index == round(corpus_size * 5 / 100, 0):
        logger.info("Processed 5% of documents")
    elif index == round(corpus_size * 10 / 100, 0):
        logger.info("Processed 10% of documents")
    elif index == round(corpus_size * 25 / 100, 0):
        logger.info("Processed 25% of documents")
    elif index == round(corpus_size * 50 / 100, 0):
        logger.info("Processed 50% of documents")
    elif index == round(corpus_size * 75 / 100, 0):
        logger.info("Processed 75% of documents")
    elif index == round(corpus_size * 100 / 100, 0):
        logger.info("Processed 100% of documents")


def count_stopwords_ratio(word_list, language):
    """
    Count stopwords ration dividing total
    number of stop words by total number
    of words in document
    :param word_list: document word list
    :param language: data set language
    :return: number of stopwords
    """
    stop_word_list = get_sw_list(language)
    detected_stop_words = [word for word in word_list if word.lower() in stop_word_list]
    logger.info(str.format('Detected Stop Words: {0}', detected_stop_words))
    logger.info(str.format('Number of stop words: {0}', len(detected_stop_words)))
    stop_word_ratio = round(len(detected_stop_words) / len(word_list), 2)
    logger.info(str.format('Stopwords ratio:  {0}', stop_word_ratio))
    return stop_word_ratio


def create_tags_dict(tags):
    tagged_tokens_dict = {}
    for tag in tags:
        splitted = tag.split('\t')
        tag = splitted[1]
        if tag in tagged_tokens_dict:
            tagged_tokens_dict[tag] += 1
        else:
            tagged_tokens_dict[tag] = 1
    return tagged_tokens_dict


def map_en_pos_tag(tag):
    if tag in NOUNS_EN:
        return 'noun'
    elif tag in VERBS_EN:
        return 'verb'
    elif tag in ADJECTIVES_EN:
        return 'adjective'
    elif tag in ADVERBS_EN:
        return 'adverb'
    elif tag in PREPOSITIONS_EN:
        return 'preposition'
    elif tag in PRONOUNS_EN:
        return 'pronoun'
    elif tag in DETERMINERS_EN:
        return 'determiner'
    elif tag in PARTICLES_EN:
        return 'particle'
    elif tag in FOREIGN_WORDS_EN:
        return 'foreign'


def map_es_pos_tag(tag):
    if tag in NOUNS_ES:
        return 'noun'
    elif tag in VERBS_ES:
        return 'verb'
    elif tag in ADJECTIVES_ES:
        return 'adjective'
    elif tag in ADVERBS_ES:
        return 'adverb'
    elif tag in ADPOSITIONS_ES:
        return 'adposition'
    elif tag in PRONOUNS_ES:
        return 'pronoun'
    elif tag in DETERMINERS_ES:
        return 'determiner'
    elif tag in FOREIGN_WORDS_EN:
        return 'foreign'


def replace_words_through_tags(tags, language):
    pos_tokens = []
    for index, tag in enumerate(tags):
        splitted_tag = ""
        splitted = tag.split('\t')
        if language is "en":
            splitted_tag = map_en_pos_tag(splitted[1])
        elif language is "es":
            splitted_tag = map_es_pos_tag(splitted[1])

        if splitted_tag is None:
            splitted_tag = "other"
        pos_tokens.append(splitted_tag)
    return ' '.join(pos_tokens)


def get_language_pos_vocabulary(language):
    if language is "en":
        return ['noun', 'verb', 'adjective', 'adverb', 'preposition', 'pronoun', 'determiner', 'particle', 'foreign']
    elif language is "es":
        return ['noun', 'verb', 'adjective', 'adverb', 'adposition', 'pronoun', 'determiner', 'foreign']


def count_capitalized_sent_ratio(sentences, total_sent):
    """
    Count percentage of sentences, which start with
    capital letter

    :param sentences:  list of sentences
    :param total_sent: total number of sentences
    :return: percentage of sentences
    """
    nmb_of_capital_cent = 0
    for sentence in sentences:
        if sentence[0].isupper():
            nmb_of_capital_cent += 1
    ratio = round((nmb_of_capital_cent * 100) / total_sent, 2)
    logger.info(str.format("Capitalized sentences ratio: {0}", ratio))
    return ratio


def count_sentence_final_token_percentage(sentences, final_token, total_sent):
    """
    Count percentage of sentences, which as last
    character have the 'final token'.

    :param sentences: list of sentences
    :param final_token: pattern for final token
    :param total_sent: total number of sentences
    :return: percentage of sentences
    """
    total = 0
    for sentence in sentences:
        if sentence[len(sentence) - 1] == final_token:
            total += 1
    ratio = round(((total * 100) / total_sent), 2)
    logger.info(str.format("Percentage of sentences, which end with {0} is {1} %", final_token, ratio))
    return ratio


def count_average_word_length(total_chars, total_words):
    """
    Count average word length with  deviding the total number
    of characters in document by total number of words in document

    :param total_chars: total number of characters
    :param total_words: total number of words
    :return: document average length
    """
    average_word_length = round((total_chars / total_words), 2)
    logger.info(str.format("Average word length: {0}", average_word_length))
    return average_word_length


def count_average_sentence_length_in_chars(total_chars, total_sent):
    """
    Count average sentence length  in characters with
    deviding the total number of characters in
    document by total number of sentences in document

    :param total_chars: total number of characters
    :param total_sent: total number of sentences
    :return: document average sentence length in characters
    """
    avg_sent_length = total_chars / total_sent
    logger.info(str.format("Average sentence length in chars: {0}", round(avg_sent_length, 2)))
    return round(avg_sent_length, 2)


def count_avg_sent_length_in_words(tokens, total_sent):
    """
    Count average sentence length  in words with
    deviding the total number of words in
    document by total number of sentences in document

    :param total_chars: total number of words
    :param total_sent: total number of sentences
    :return: document average sentence length in words
    """
    avg_sent_length = len(tokens) / total_sent
    logger.info(str.format("Average sentence length in words: {0}", round(avg_sent_length, 2)))
    return round(avg_sent_length, 2)


def count_capital_letters(tokens, total_chars):
    """
    Count capital letters' ratio in document

    :param tokens: list of tokens
    :param total_chars: total number of characters
    :return: capital letters' ratio
    """
    total_capital_letters = (sum(1 for c in ''.join(tokens) if c.isupper()))
    ratio = round((total_capital_letters / total_chars), 2)
    logger.info(str.format("Capital letters ratio: {0}", ratio))
    return ratio


def count_capital_tokens(tokens):
    """
    Count capital tokens' ratio in document

    :param tokens: list of tokens
    :return: capital tokens' ratio
    """
    total_capitalized = 0
    for token in tokens:
        if token[0].isupper():
            total_capitalized += 1
    ratio = round((total_capitalized / len(tokens)), 2)
    logger.info(str.format("Capital tokens ratio: {0}", ratio))
    return ratio


def count_total_chars(document_word_list):
    """
    Count total number of characters in the text

    :param document_word_list:  list of document words
    :return: total number of characters
    """
    text = ''.join(document_word_list)
    text = strip_white_spaces(text)
    text = strip_apostroph(text)
    total_chars = len(text)
    logger.info(str.format("Number of characters: {0}", total_chars))
    return total_chars


def count_avg_digits(text, total_characters):
    """
    Count  number of digits in document
    by dividing number of digits by
    total number of characters in document
    :param text: document text
    :param total_characters: number of characters in document
    :return: number (float) of digits in input document text
    """
    total_digits = len([char for char in text if char.isdigit()])
    result = round((total_digits / total_characters), 2)
    logger.info(str.format("Number of digits: {0}", result))
    return result


def count_punctuation_ratio(text, punctuation_mark, total_centences, msg):
    """
    Count number of input punctuation mark
    dividing number of some punctuation marks by
    the total number of sentences in the document
    :param text: document text
    :param punctuation_mark: punctuation mark
    :param total_characters: ttal number of characters
    :return: punctuation mark ratio
    """
    punctuation_mark_sum = text.count(punctuation_mark)
    ratio = round((punctuation_mark_sum / total_centences), 2)
    logging.info(str.format("{0}{1}", msg, ratio))
    return ratio


def preprocess_text_for_punct_count(text):
    """
    Preprocess input text

    :param text:  unprocessed text
    :return: preprocessed text
    """
    #text = remove_tabs_and_newlines(text)
    #text = remove_urls(text)
    text = strip_accent_mark(text)
    text = remove_multiple_spaces(text)
    return text


def preprocess_text_for_vocabulary_richness(text):
    text = alter_apostroph_prefix(text)
    text = strip_accent_mark(text)
    text = strip_non_latin(text)
    text = strip_punctuation(text)
    text = remove_digits(text)
    text = remove_multiple_spaces(text)
    return text


def get_unique_words(tokens):
    unique_words = []
    for token in tokens:
        if token not in unique_words:
            unique_words.append(token)
    return unique_words


def build_list_dict(word_list):
    word_list_dict = {}
    for word in word_list:
        if word not in word_list_dict.keys():
            word_list_dict[word] = 1
        else:
            appear = word_list_dict[word]
            word_list_dict[word] = appear + 1
    return word_list_dict


def count_legomenon(word_list_dict, number):
    counter = 0
    for word, appearance in word_list_dict.items():
        if appearance == number:
            counter = counter + 1
    return counter


def count_type_token_ratio(unique_words_total, words_total):
    ttr = unique_words_total / words_total * 100
    return round(ttr, 2)


# #Honore Measure =(100*logN) / (1-V_1/V)
def count_honore_measure(word_list_dict, words_total, unique_words_total):
    hapax_legomenon_of_length = count_legomenon(word_list_dict, 1)
    r = (100 * math.log10(words_total)) / (1 - hapax_legomenon_of_length / unique_words_total)
    return round(r, 2)


# Source: https://edu.cs.uni-magdeburg.de/EC/lehre/sommersemester-2013/wissenschaftliches-schreiben-in-der-informatik/publikationen-fuer-studentische-vortraege/HooverAnotherPerspective.pdf
def count_sichel_measure(word_list_dict, unique_words_total):
    total_dis_legomenon = count_legomenon(word_list_dict, 2)
    s = total_dis_legomenon / unique_words_total
    return round(s, 2)


# # Yule Measure K = 10000*(M-N) / N^2;     M = sum (i^2 * V_i)
# # http://pers-www.wlv.ac.uk/~in4326/papers/$U50.pdf
def count_yule_measure(word_list_dict, words_total):
    m = 0
    unique_values = set(word_list_dict.values())
    for value in unique_values:
        m = m + pow(value, 2) * count_legomenon(word_list_dict, value)
    k = (10000 * (m - words_total)) / pow(words_total, 2)
    return round(k, 2)


def count_syllables_es(word):
    word = re.sub(r'\W+', '', word)
    syllables = silabizer()
    return len(syllables(word))


def count_total_syllables(tokens):
    total_syllables = 0
    for token in tokens:
        total_syllables += count_syllables_es(token)
        # print(token, new, total_snippet_syllables)
    return total_syllables


def get_text_snippet(full_text, word_limit):
    splitted = full_text.split()
    start_index = (randint(0, len(splitted)))
    snippet_word_list = splitted[start_index:start_index + (word_limit - 1)]
    if len(snippet_word_list) < word_limit:
        differ = word_limit - len(snippet_word_list)
        rest = splitted[0:start_index]
        additional_word_list = rest[0:differ]
        for word in additional_word_list:
            snippet_word_list.append(word)
    #print("Size of snippet word list", len(snippet_word_list))
    return ' '.join(snippet_word_list)


def fernandez_huerta_index(total_syllables, total_words, total_sentences):
    # IFH = 206.84 - (0.60 * total_syllables) - (1.02 * total_sentences)
    ifh = 206.84 - (60 * (total_syllables / total_words)) - (1.02 * total_words / total_sentences)
    print('Fernandez Huerta Index: ', round(ifh, 2))
    return round(ifh, 2)


def flesch_szigriszt_index(total_syllables, total_words, total_sentences):
    ifs = 206.835 - (62.3 * (total_syllables / total_words)) - (total_words / total_sentences)
    print('Flesch Szigriszt Index: ', round(ifs, 2))
    return round(ifs, 2)


def crawford_readability(total_syllables, total_words, total_sentences):
    """ Comprensibilidad de Gutiérrez de Polini = 95,2 – (9,7 x L/P) – (0,35 x P/F)
            Crawford's readability formula = -0,205OP + 0,049SP – 3,407
            OP = mean number of sentences per 100 words
            SP = mean number of syllables in 100 words"""
    op = total_sentences / total_words * 100
    sp = total_syllables / total_words * 100
    cr = -0.205 * op + 0.049 * sp - 3.407
    print('Crawford Readability (years): ', round(cr, 1))
    return round(cr, 1)


def gutierrez_de_poloni_score(total_letters, total_words, total_sentences):
    """Gutiérrez de Polini's readability score (1972)"""
    GPS = 95.2 - 9.7 * (total_letters / total_words) - 0.35 * (total_words / total_sentences)
    print('Gutiérrez de Polinis readability score: ', round(GPS, 2))
    return round(GPS, 2)


def preprocess_text_snippet(text):
    text = alter_apostroph_prefix(text)
    text = strip_accent_mark(text)
    text = strip_non_latin(text)
    text = strip_punctuation(text)
    text = remove_digits(text)
    text = remove_multiple_spaces(text)
    return text



#!/usr/bin/env python3
# Based on Mabodo's ipython notebook (https://github.com/mabodo/sibilizador)
# (c) Mabodo
# CODE QUELLE: https://github.com/amunozf/separasilabas/blob/master/separasilabas.py

class char():
    def __init__(self):
        pass

class char_line():
    def __init__(self, word):
        self.word = word
        self.char_line = [(char, self.char_type(char)) for char in word]
        self.type_line = ''.join(chartype for char, chartype in self.char_line)

    def char_type(self, char):
        if char in set(['a', 'á', 'e', 'é', 'o', 'ó', 'í', 'ú']):
            return 'V'  # strong vowel
        if char in set(['i', 'u', 'ü']):
            return 'v'  # week vowel
        if char == 'x':
            return 'x'
        if char == 's':
            return 's'
        else:
            return 'c'

    def find(self, finder):
        return self.type_line.find(finder)

    def split(self, pos, where):
        return char_line(self.word[0:pos + where]), char_line(self.word[pos + where:])

    def split_by(self, finder, where):
        split_point = self.find(finder)
        if split_point != -1:
            chl1, chl2 = self.split(split_point, where)
            return chl1, chl2
        return self, False

    def __str__(self):
        return self.word

    def __repr__(self):
        return repr(self.word)


class silabizer():
    def __init__(self):
        self.grammar = []

    def split(self, chars):
        rules = [('VV', 1), ('cccc', 2), ('xcc', 1), ('ccx', 2), ('csc', 2), ('xc', 1), ('cc', 1), ('vcc', 2),
                 ('Vcc', 2), ('sc', 1), ('cs', 1), ('Vc', 1), ('vc', 1), ('Vs', 1), ('vs', 1)]
        for split_rule, where in rules:
            first, second = chars.split_by(split_rule, where)
            if second:
                if first.type_line in set(['c', 's', 'x', 'cs']) or second.type_line in set(['c', 's', 'x', 'cs']):
                    # print 'skip1', first.word, second.word, split_rule, chars.type_line
                    continue
                if first.type_line[-1] == 'c' and second.word[0] in set(['l', 'r']):
                    continue
                if first.word[-1] == 'l' and second.word[-1] == 'l':
                    continue
                if first.word[-1] == 'r' and second.word[-1] == 'r':
                    continue
                if first.word[-1] == 'c' and second.word[-1] == 'h':
                    continue
                return self.split(first) + self.split(second)
        return [chars]

    def __call__(self, word):
        return self.split(char_line(word))


class DocCorpus:
    def __init__(self, doc_dir_path):
        self.doc_dir_path = doc_dir_path
        self.truth_path = os.path.join(doc_dir_path, "truth.txt")
        self.documents = []

    def load(self):
        logger.info("Loading corpus of documents.....")
        xml_files = glob.glob(str.format("{0}\*.xml", self.doc_dir_path))
        truth_file_list = [row.rstrip('\n') for row in open(self.truth_path)]
        for row in truth_file_list:
            author_id, age = extract_document_label(row)
            if age is not None and age != 0:
                raw_text = get_raw_text(self.doc_dir_path, xml_files, author_id)
                if raw_text is None or raw_text == '':
                    logger.info(str.format('WARNING: Document {0} is empty.',
                                           os.path.join(self.doc_dir_path, str.format('{0}.xml', author_id))))
                else:
                    document = (author_id, age, raw_text)
                    self.documents.append(document)
        self.get_statistics()
        return self.documents

    def get_statistics(self):
        age_dict = {}
        for document in self.documents:
            age = document[1]
            if age not in age_dict:
                age_dict[age] = 1
            else:
                age_dict[age] += 1
        logger.info('DESTRIBUTION OF DOCUMENTS BY AGE:')
        logger.info("-" * 40)
        logger.info(str.format('   Age label {0}Number of documents', ' ' * 4, ))
        logger.info("-" * 40)
        logger.info(str.format('     18-24{0}{1}', ' ' * 13, age_dict.get("1")))
        logger.info(str.format('     25-34{0}{1}', ' ' * 13, age_dict.get("2")))
        logger.info(str.format('     35-49{0}{1}', ' ' * 13, age_dict.get("3")))
        logger.info(str.format('     50-64{0}{1}', ' ' * 13, age_dict.get("4")))
        logger.info(str.format('     65-xx{0}{1}', ' ' * 13, age_dict.get("5")))
        total = age_dict.get("1") + age_dict.get("2") + age_dict.get("3") + age_dict.get("4") + age_dict.get("5")
        logger.info(str.format('     Total {0}{1}', ' ' * 12, total))
        logger.info("-" * 40)


#  Features
class NGramms:
    def __init__(self, feature_collection_dir, language, document_corpus, ):
        self.document_corpus = document_corpus
        self.language = language
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.author_id_list = []
        self.label_list = []
        self.corpus_dict = {}
        self.feature_collection_dir = feature_collection_dir

    def import_language_class(self):
        if self.language == "english":
            return English()
        elif self.language == "spanish":
            return Spanish()

    def compute(self, language, n, min_df, max_df, max_features):
        nlp = self.import_language_class()
        if arguments.text_preprocessing.lower() == "lemma":
            logging.info(str.format("Loading spacy model"))
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            #logging.info(str.format("Author_ID: {0}", document[0]))
            self.label_list.append(document[1])
            text = preprocess_text_for_ngramms(document[2], self.language)
            bag_of_words = get_bow(text)
            # remove words, which have length <3
            bag_of_words = [w for w in bag_of_words if len(w) > 2]


            if arguments.text_preprocessing.lower() == "lemma":
                bag_of_words = lemmatise_tokens(bag_of_words, nlp)
                # delete digits
                bag_of_words = [x for x in bag_of_words if not (x.isdigit()
                                                                or x[0] == '-' and x[1:].isdigit())]
                #logging.info(str.format("Lemmas: {0}\n", [x.encode('utf-8') for x in bag_of_words]))

            elif arguments.text_preprocessing.lower() == "stemm":
                bag_of_words = stemm_tokens(language, bag_of_words)
                #logging.info(str.format("Stemms: {0}\n", [x.encode('utf-8') for x in bag_of_words]))
            self.corpus_dict[document[0]] = ' '.join(bag_of_words)

        logger.info("Vectorizing")
        tf_idf = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, ngram_range=(n, n),
                                 stop_words=get_sw_list(language), norm=None)
        features = tf_idf.fit_transform(self.corpus_dict.values())
        feature_names = tf_idf.get_feature_names()
        df = pd.DataFrame(np.round(features.toarray(), 2), columns=feature_names)
        df.insert(loc=0, column='lbl', value=self.label_list)
        df.insert(loc=0, column='author_id', value=self.author_id_list)
        filename = "{0}__ngram={1}__mindf={2}__maxdf={3}__max_features={4}.csv".format(language, n, min_df, max_df,
                                                                                       max_features)
        file_path = os.path.join(self.feature_collection_dir, filename)
        df.to_csv(file_path, sep=',', index=False, encoding='utf-8')

        logger.info("Vocabulary nGrams sorted by frequency:")
        scores = zip(feature_names, np.asarray(features.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for index, element in enumerate(sorted_scores):
            if index <= 20:
                logger.info("{0:50} Score: {1}".format(element[0], element[1]))

        logger.info("Number of features: {0}".format(len(feature_names)))
        #logger.info("Top 20 feature names: {0}".format(feature_names[:20]))
        logger.info(str.format("Extracted features were saved to: {0}", file_path))


class Emoticons:
    def __init__(self, document_corpus, feature_collection_dir):
        self.document_corpus = document_corpus
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.author_id_list = []
        self.label_list = []
        self.corpus_dict = {}
        self.feature_collection_dir = feature_collection_dir

    def compute(self, min_df, max_df, max_features):
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            #logging.info(str.format("Author_id: {0}", document[0]))
            text = preprocess_text_for_emoticons(document[2])
            self.corpus_dict[document[0]] = text

        logger.info("Vectorizing")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), vocabulary=EMOTICON_DICT)
        emoticon_features = vectorizer.fit_transform(self.corpus_dict.values())
        emoticon_feature_names = vectorizer.get_feature_names()

        df = pd.DataFrame(np.round(emoticon_features.toarray(), 2), columns=emoticon_feature_names)
        df.insert(loc=0, column='lbl', value=self.label_list)
        df.insert(loc=0, column='author_id', value=self.author_id_list)
        filename = "emoticons__mindf={0}__maxdf={1}__max_features={2}.csv".format(min_df, max_df, max_features)
        file_path = os.path.join(self.feature_collection_dir, filename)
        df.to_csv(file_path, sep=',', index=False, encoding='utf-8')


        logger.info("Vocabulary nGrams sorted by frequency:")
        scores = zip(emoticon_feature_names, np.asarray(emoticon_features.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for index, element in enumerate(sorted_scores):
                logger.info("{0:50} Score: {1}".format(element[0], element[1]))

        logger.info("Number of features: {0}".format(len(emoticon_feature_names)))
        #logger.info("Top 20 feature names: {0}".format(feature_names[:20]))
        logger.info(str.format("Extracted features were saved to: {0}", file_path))



class MisspellRatio:
    def __init__(self, language, document_corpus, feature_collection_dir):
        self.language = language
        self.document_corpus = document_corpus
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.author_id_list = []
        self.label_list = []
        self.misspell_density_list = []
        self.feature_collection_dir = feature_collection_dir

    def compute(self):
        feature_collection = pd.DataFrame()
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            logging.info(str.format("Author_id: {0}", document[0]))
            text = preprocess_text_for_bow(document[2])



            document_tokens = word_tokenize(text, self.language)
            unknown_tokens = check_spelling(document_tokens, self.language)
            misspelled_tokens = get_misspelled_tokens(unknown_tokens, self.language)
            misspell_ratio = count_ratio(len(document_tokens), len(misspelled_tokens))
            self.misspell_density_list.append(misspell_ratio)
        feature_collection['misspell_ratio'] = self.misspell_density_list

        file_path = os.path.join(self.feature_collection_dir, "misspell_ratio.csv")
        feature_collection.insert(loc=0, column='lbl', value=self.label_list)
        feature_collection.insert(loc=0, column='author_id', value=self.author_id_list)
        feature_collection.to_csv(file_path, sep=',', encoding='utf-8', index=False)
        logger.info(str.format("Extracted features were saved to: {0}", file_path))


class StopwordsRatio:
    def __init__(self, language, document_corpus, feature_collection_dir):
        self.document_corpus = document_corpus
        self.language = language
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.total_characters = 0
        self.author_id_list = []
        self.label_list = []
        self.stopwords_ratio = []
        self.feature_collection_dir = feature_collection_dir

    def compute(self):
        """
         create new 'feature collection' Dataframe
        of 'stopwords_ratio' feature and save it to csv-file.
        """
        feature_collection = pd.DataFrame()
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            logging.info(str.format("Author_id: {0}", document[0]))
            text = preprocess_text_for_bow(document[2])
            print(text)
            document_tokens = text.split()
            document_tokens = [token.lower() for token in document_tokens]
            self.stopwords_ratio.append(count_stopwords_ratio(document_tokens, self.language))
            logger.info("-" * 60)
        # Add add feature values to feature collection file
        feature_collection['stopwords_ratio'] = self.stopwords_ratio
        # Save features
        file_path = os.path.join(self.feature_collection_dir, "stopwords_ratio.csv")
        feature_collection.insert(loc=0, column='lbl', value=self.label_list)
        feature_collection.insert(loc=0, column='author_id', value=self.author_id_list)
        feature_collection.to_csv(file_path, sep=',', encoding='utf-8', index=False)
        logger.info(str.format("Extracted features were saved to: {0}", file_path))


class PosCounter:
    def __init__(self, language, document_corpus, feature_collection_dir):
        self.language = get_lang(language)
        self.document_corpus = document_corpus
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.author_id_list = []
        self.label_list = []
        self.corpus_dict = {}
        self.feature_collection_dir = feature_collection_dir

    def compute(self):
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            #logging.info(str.format("Author_id: {0}", document[0]))
            text = preprocess_text_for_bow(document[2])

            tagger = treetaggerwrapper.TreeTagger(TAGLANG=self.language, TAGDIR="C:/TreeTagger")
            tagged_text = replace_words_through_tags(tagger.tag_text(text), self.language)

            self.corpus_dict[document[0]] = tagged_text

        logger.info("Vectorizing")
        vectorizer = TfidfVectorizer(vocabulary=get_language_pos_vocabulary(self.language))
        pos_features = vectorizer.fit_transform(self.corpus_dict.values())
        pos_features_names = vectorizer.get_feature_names()

        feature_collection = pd.DataFrame(np.round(pos_features.toarray(), 2), columns=pos_features_names)
        feature_collection.insert(loc=0, column='lbl', value=self.label_list)
        feature_collection.insert(loc=0, column='author_id', value=self.author_id_list)
        filename = "pos_features.csv"
        file_path = os.path.join(self.feature_collection_dir, filename)
        feature_collection.to_csv(file_path, sep=',', index=False, encoding='utf-8')


        logger.info("Vocabulary nGrams sorted by frequency:")
        scores = zip(pos_features_names, np.asarray(pos_features.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for index, element in enumerate(sorted_scores):
                logger.info("{0:50} Score: {1}".format(element[0], element[1]))

        logger.info("Number of features: {0}".format(len(pos_features_names)))
        #logger.info("Feature names: {0}".format(pos_features_names))
        logger.info(str.format("Extracted features were saved to: {0}", file_path))


class LexicalFeatures:
    """
    LexicalFeatures class counts lexical features and
    saves they to the CSV-file in a Matrix form.
    """

    def __init__(self, language, document_corpus, feature_collection_dir):
        self.language = language
        self.feature_collection_dir = feature_collection_dir
        self.document_corpus = document_corpus
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.author_id_list = []
        self.label_list = []
        self.total_sentences = []
        self.capital_sent_percentage = []
        self.declarative_sent_percentage = []
        self.question_sent_percentage = []
        self.exclamation_sent_percentage = []
        self.avg_sent_length_in_chars = []
        self.avg_sent_length_in_words = []
        self.avg_word_length = []
        self.avg_digits = []
        self.capital_letters = []
        self.capital_tokens = []

    def compute(self):
        """
         create new 'feature collection' Dataframe
        and save it to csv-file.
        """
        feature_collection = pd.DataFrame()
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            text = document[2]
            # Extract features
            self.extract_features(text)
        # Add 'white_spaces' and 'digits' column to feature collection file
        feature_collection['capital_sent_percentage'] = self.capital_sent_percentage
        feature_collection['declarative_sent_percentage'] = self.declarative_sent_percentage
        feature_collection['question_sent_percentage'] = self.question_sent_percentage
        feature_collection['exclamation_sent_percentage'] = self.exclamation_sent_percentage
        feature_collection['avg_sent_length_in_chars'] = self.avg_sent_length_in_chars
        feature_collection['avg_sent_length_in_words'] = self.avg_sent_length_in_words
        feature_collection['avg_word_length'] = self.avg_word_length
        feature_collection['avg_digits'] = self.avg_digits
        feature_collection['capital_letters'] = self.capital_letters
        feature_collection['capital_tokens'] = self.capital_tokens

        # Save features
        file_path = os.path.join(self.feature_collection_dir, "lexical_features.csv")
        feature_collection.insert(loc=0, column='lbl', value=self.label_list)
        feature_collection.insert(loc=0, column='author_id', value=self.author_id_list)
        feature_collection.to_csv(file_path, sep=',', index=False, encoding='utf-8')
        logger.info(str.format("Extracted features were saved to: {0}", file_path))

    def extract_features(self, text):
        """
        Extract features from input text
        :param text: document text
        """
        #text = remove_urls(text)
        sentence_list = sent_tokenize(text, self.language)
        self.total_sent = len(sentence_list)
        self.total_sentences.append(self.total_sent * 1.0)

        self.capital_sent_percentage.append(count_capitalized_sent_ratio(sentence_list, self.total_sent))
        self.declarative_sent_percentage.append(
            count_sentence_final_token_percentage(sentence_list, '.', self.total_sent))
        self.question_sent_percentage.append(count_sentence_final_token_percentage(sentence_list, '?', self.total_sent))
        self.exclamation_sent_percentage.append(
            count_sentence_final_token_percentage(sentence_list, '!', self.total_sent))

        #text = remove_tabs_and_newlines(text)
        text = alter_apostroph_prefix(text)
        text = strip_punctuation(text)

        total_chars_with_digits = count_total_chars(text.split())
        self.avg_digits.append(count_avg_digits(text, total_chars_with_digits))
        text = remove_digits(text)
        doc_tokens = text.split()
        total_chars_without_digits = count_total_chars(doc_tokens)
        avg_sent_length_in_chars = count_average_sentence_length_in_chars(total_chars_without_digits, self.total_sent)

        self.avg_sent_length_in_words.append(count_avg_sent_length_in_words(doc_tokens, self.total_sent))
        self.avg_sent_length_in_chars.append(avg_sent_length_in_chars)
        self.avg_word_length.append(count_average_word_length(total_chars_without_digits, len(doc_tokens)))
        self.capital_letters.append(count_capital_letters(doc_tokens, total_chars_without_digits))
        self.capital_tokens.append(count_capital_tokens(doc_tokens))


class PunctuationCounter:
    """
    Count how often the specific punctuation
    appears in the document
    """

    def __init__(self, language, document_corpus, feature_collection_dir):
        self.document_corpus = document_corpus
        self.feature_collection_dir = feature_collection_dir
        self.language = language
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.total_sent = 0
        self.author_id_list = []
        self.label_list = []
        self.comma_list = []
        self.mult_dot_list = []
        self.mult_exclamation_list = []
        self.mult_question_list = []
        self.colon_list = []
        self.semicolon_list = []

    def compute(self):
        """
        create new 'feature collection' Dataframe
        of puctuation features and save it to csv-file.
        """
        feature_collection = pd.DataFrame()
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            logging.info(str.format("Autor_id: {0}", document[0]))
            text = document[2]
            text = preprocess_text_for_punct_count(text)
            # Count number of sentences
            sentence_list = sent_tokenize(text, self.language)
            self.total_sent = len(sentence_list)
            # Extract features
            self.extract_features(text)
        # Add add feature values to feature collection file
        feature_collection['comma_ratio'] = self.comma_list
        feature_collection['mult_dot_ratio'] = self.mult_dot_list
        feature_collection['mult_exclamation_ratio'] = self.mult_exclamation_list
        feature_collection['mult_question_ratio'] = self.mult_question_list
        feature_collection['colon_ratio'] = self.colon_list
        feature_collection['semicolon_ratio'] = self.semicolon_list

        # Save features
        file_path = os.path.join(self.feature_collection_dir, "punctuation_features.csv")
        feature_collection.insert(loc=0, column='lbl', value=self.label_list)
        feature_collection.insert(loc=0, column='author_id', value=self.author_id_list)
        feature_collection.to_csv(file_path, sep=',', index=False, encoding='utf-8')
        logger.info(str.format("Extracted features were saved to: {0}", file_path))

    def extract_features(self, text):
        """
        Extract from input text number of different
        punctuation characters and add these values to the
        feature collection matrix
        :param text: document test
        """
        self.comma_list.append(
            count_punctuation_ratio(text, ',', self.total_sent, "Average number of commas in sentence: "))
        self.mult_dot_list.append(
            count_punctuation_ratio(text, '...', self.total_sent, "Average number of multiple puncts in sentence: "))
        self.mult_exclamation_list.append(
            count_punctuation_ratio(text, '!!!', self.total_sent,
                                    "Average number of multiple exclamations in sentence: "))
        self.mult_question_list.append(count_punctuation_ratio(
            text, '???', self.total_sent, "Average number of multiple question marks in sentence: "))
        self.colon_list.append(
            count_punctuation_ratio(text, ':', self.total_sent, "Average number of colons in sentence: "))
        self.semicolon_list.append(
            count_punctuation_ratio(text, ';', self.total_sent, "Average number of semicolons in sentence: "))


class VocabularyRichness:
    def __init__(self, language, document_corpus, feature_collection_dir):
        self.document_corpus = document_corpus
        self.feature_collection_dir = feature_collection_dir
        self.language = language
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.author_id_list = []
        self.label_list = []
        self.type_token_ratio = []
        self.honore_measure = []
        self.sichel_measure = []
        self.yule_measure = []

    def compute(self):
        feature_collection = pd.DataFrame()
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            print(document[0]) #todo: Delete
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            text = preprocess_text_for_vocabulary_richness(document[2])
            self.extract_features(text)

        feature_collection['type_token_ratio'] = self.type_token_ratio
        feature_collection['honore_measure'] = self.honore_measure
        feature_collection['sichel_measure'] = self.sichel_measure
        feature_collection['yule_measure'] = self.yule_measure

        # Save features
        file_path = os.path.join(self.feature_collection_dir, "vocabulary_richness.csv")
        feature_collection.insert(loc=0, column='lbl', value=self.label_list)
        feature_collection.insert(loc=0, column='author_id', value=self.author_id_list)
        feature_collection.to_csv(file_path, sep=',', index=False, encoding='utf-8')
        logger.info(str.format("Extracted features were saved to: {0}", file_path))

    def extract_features(self, text):
        document_tokens = text.split()
        document_tokens = [token.lower() for token in document_tokens]
        words_total = len(document_tokens)
        unique_words = get_unique_words(document_tokens)
        word_list_dict = build_list_dict(document_tokens)

        self.type_token_ratio.append(count_type_token_ratio(len(unique_words), words_total))
        self.honore_measure.append(count_honore_measure(word_list_dict, words_total, len(unique_words)))
        self.sichel_measure.append(count_sichel_measure(word_list_dict, len(unique_words)))
        self.yule_measure.append(count_yule_measure(word_list_dict, words_total))


class ReadabilityScores:
    def __init__(self, language, document_corpus, feature_collection_dir):
        self.feature_collection_dir = feature_collection_dir
        self.language = language
        self.document_corpus = document_corpus
        self.labels = '18-24,18-24,35-49,50-64,65-xx'
        self.author_id_list = []
        self.label_list = []

        self.auto_read_index = []
        self.flesh_reading_ease = []
        self.flesch_kincaid_grade = []
        self.gunning_fog = []

        self.fernandez_huerta_index = []
        self.flesch_szigriszt_index = []
        self.crawford_readability = []
        self.gutierrez_de_poloni_score = []

    def compute(self):
        feature_collection = pd.DataFrame()
        for index, document in enumerate(self.document_corpus):
            log_processing_progress(index, len(self.document_corpus))
            self.author_id_list.append(document[0])
            self.label_list.append(document[1])
            text = strip_multiple_whitespaces(document[2])
            total_sentences = len(sent_tokenize(text, self.language))
            total_spaces = text.count(" ")
            total_words = total_spaces + 1
            if self.language == 'english':
                self.count_readability_english(text)
            if self.language == 'spanish':
                self.count_readability_spanish(text, total_words, total_sentences)

        if self.language == 'english':
            feature_collection['auto_read_index'] = self.auto_read_index
            feature_collection['flesh_reading_ease'] = self.flesh_reading_ease
            feature_collection['flesch_kincaid_grade'] = self.flesch_kincaid_grade
            feature_collection['gunning_fog'] = self.gunning_fog

        elif self.language == 'spanish':
            feature_collection['fernandez_huerta_index'] = self.fernandez_huerta_index
            feature_collection['flesch_szigriszt_index'] = self.flesch_szigriszt_index
            feature_collection['crawford_readability'] = self.crawford_readability
            feature_collection['gutierrez_de_poloni_score'] = self.gutierrez_de_poloni_score

        # Save features
        file_path = os.path.join(self.feature_collection_dir, "readability_scores.csv")
        feature_collection.insert(loc=0, column='lbl', value=self.label_list)
        feature_collection.insert(loc=0, column='author_id', value=self.author_id_list)
        feature_collection.to_csv(file_path, sep=',', index=False, encoding='utf-8')
        logger.info(str.format("Extracted features were saved to: {0}", file_path))

    def count_readability_english(self, text):
        self.auto_read_index.append(textstat.automated_readability_index(text))  # Score[1-14]
        self.flesh_reading_ease.append(textstat.flesch_reading_ease(text))  # Score[0-100]
        self.flesch_kincaid_grade.append(textstat.flesch_kincaid_grade(text))
        self.gunning_fog.append(textstat.gunning_fog(text))

    def count_readability_spanish(self, text, total_words, total_sentences):
        text_snippet = get_text_snippet(text, 100)
        # print('Text snippet: ', text_snippet)

        sentences = sent_tokenize(text_snippet, self.language)
        total_snippet_sentences = len(sentences)
        text_snippet = preprocess_text_snippet(text_snippet)
        tokens = text_snippet.split(" ")
        # print('Snippet tokens:', tokens)

        total_snippet_syllables = count_total_syllables(tokens)
        total_letters = len(''.join(tokens))

        self.fernandez_huerta_index.append(fernandez_huerta_index(
            total_snippet_syllables, len(tokens), total_snippet_sentences))
        self.flesch_szigriszt_index.append(flesch_szigriszt_index(
            total_snippet_syllables, len(tokens), total_snippet_sentences))
        self.crawford_readability.append(crawford_readability(
            total_snippet_syllables, len(tokens), total_snippet_sentences))
        self.gutierrez_de_poloni_score.append(
            gutierrez_de_poloni_score(total_letters, total_words, total_sentences))


class FeaturesExtractor:
    def __init__(self):
        self.language = arguments.language
        self.feature_name = arguments.feature_name
        #self.csv_path = compile_file_path(self.language)

    def generate_features_dataset(self, doc_dir, feature_collection_dir):
        corpus = DocCorpus(doc_dir)
        document_corpus = corpus.load()
        self.feature_collection_dir = feature_collection_dir
        self.count_features(document_corpus)


    def count_features(self, document_corpus):
        if self.feature_name == 'n_grams':
            ngrams = NGramms(self.feature_collection_dir, self.language, document_corpus)
            return ngrams.compute(arguments.language, n=arguments.ngram, min_df=arguments.min_df,
                                  max_df=arguments.max_df, max_features=arguments.max_features)
        elif self.feature_name == 'emoticons':
            emoticons = Emoticons(document_corpus, self.feature_collection_dir)
            emoticons.compute(min_df=arguments.min_df, max_df=arguments.max_df, max_features=arguments.max_features)
        elif self.feature_name == 'misspell_ratio':
            mr = MisspellRatio(self.language, document_corpus, self.feature_collection_dir)
            return mr.compute()
        elif self.feature_name == 'stopwords_ratio':
            sw_ratio = StopwordsRatio(self.language, document_corpus, self.feature_collection_dir)
            sw_ratio.compute()
        elif self.feature_name == 'pos_features':
            pos_counter = PosCounter(self.language, document_corpus, self.feature_collection_dir)
            pos_counter.compute()
        elif self.feature_name == 'lexical_features':
            lf = LexicalFeatures(self.language, document_corpus, self.feature_collection_dir)
            lf.compute()
        elif self.feature_name == 'punctuation_features':
            pc = PunctuationCounter(self.language, document_corpus, self.feature_collection_dir)
            pc.compute()
        elif self.feature_name == 'vocabulary_richness':
            vocab_richness = VocabularyRichness(self.language, document_corpus, self.feature_collection_dir)
            return vocab_richness.compute()
        elif self.feature_name == 'readability_scores':
            read_score = ReadabilityScores(self.language, document_corpus, self.feature_collection_dir)
            return read_score.compute()
        else:
            raise ValueError("Unknown feature name")


def create_feature_collection_dir(project_root_dir):
    root_dir = os.path.join(project_root_dir, "Feature-Collection")
    language_dir = os.path.join(root_dir, arguments.language)
    dataset_dir = os.path.join(language_dir, arguments.dataset_type)
    create_dir(root_dir)
    create_dir(language_dir)
    create_dir(dataset_dir)
    return dataset_dir


if __name__ == "__main__":
    start = time.time()
    arguments = config_arg_parser()
    project_root_dir = os.path.join(os.path.expanduser("~/Desktop"), "Age-Detection")
    log_dir = os.path.join(project_root_dir, "Log")
    create_dir(project_root_dir)
    create_dir(log_dir)
    log_file = os.path.join(log_dir, 'feature_extraction_{0}__{1}.log'.format(arguments.feature_name, strftime("%Y-%m-%d__%H-%M-%S", gmtime())))
    init_logging(log_dir, log_file)
    log_dataset_description()
    logger.info(str.format("Started feature extraction ....."))

    # create directory to save features
    feature_collection_dir = create_feature_collection_dir(project_root_dir)

    db = FeaturesExtractor()
    db.generate_features_dataset(arguments.doc_dir, feature_collection_dir)

    logger.info(str.format("Log file path: {0}", log_file))
    logger.info("Feature extraction finished")
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Total execution time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
