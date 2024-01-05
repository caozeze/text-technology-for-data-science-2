import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from decimal import Decimal
from nltk.stem import SnowballStemmer
import re
from xml.etree import ElementTree as ET
import statistics
from collections import defaultdict
from gensim import corpora, models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.sparse import dok_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def ir_eval(sys_results, qrels):
    """
    Calculate the following metrics for each system and query:
    - P@10: precision at cutoff 10 (only top 10 retrieved documents in the list are considered for each query).
    - R@50: recall at cutoff 50 (only top 50 retrieved documents in the list are considered for each query).
    - R-precision: precision at cutoff R (R is the number of relevant documents for a given query).
    - AP: average precision - hint: for all previous scores, the value of relevance should be considered as 1.
    - nDCG@10: normalized discount cumulative gain at cutoff 10.
    - nDCG@20: normalized discount cumulative gain at cutoff 20.

    Parameters:
    sys_results (pd.DataFrame): A dataframe containing the system results.
    qrels (pd.DataFrame): A dataframe containing the qrels.

    Returns:
    pd.DataFrame: A dataframe containing the evaluation results.
    """
    sys_numbers = sys_results['system_number'].unique()
    query_numbers = sys_results['query_number'].unique()
    
    ir_eval = []
    for sys_num in sys_numbers:
        for query_num in query_numbers:
            current_list = []
            # Get the current system and query results
            current_sys_df = sys_results.loc[(sys_results['system_number'] == sys_num) & (sys_results['query_number'] == query_num)]
            current_sys_df = current_sys_df.sort_values(by=['rank_of_doc']).reset_index(drop=True)
            current_qresl_df = qresl.loc[qresl['query_id'] == query_num].sort_values(by=['relevance'],ascending=False).reset_index(drop=True)
            current_qresl_doc_numbers = set(current_qresl_df['doc_id'].tolist())
            
            # P@10: precision at cutoff 10 (only top 10 retrieved documents in the list are considered for each query). 
            p_10 = calculate_p_10(current_sys_df, current_qresl_doc_numbers)

            # R@50: recall at cutoff 50 (only top 50 retrieved documents in the list are considered for each query).
            r_50 = calculate_r_50(current_sys_df, current_qresl_doc_numbers)

            # R-precision: precision at cutoff R (R is the number of relevant documents for a given query).
            r_precision = calculate_r_precision(current_sys_df, current_qresl_doc_numbers)

            # AP: average precision - hint: for all previous scores, the value of relevance should be considered as 1.
            ap = calculate_ap(current_sys_df, current_qresl_doc_numbers)

            # # nDCG@10: normalized discount cumulative gain at cutoff 10. 
            ndcg_10 = calculate_ndcg_10(current_sys_df, current_qresl_df, current_qresl_doc_numbers)
            
            # nDCG@20: normalized discount cumulative gain at cutoff 20. 
            ndcg_20 = calculate_ndcg_20(current_sys_df, current_qresl_df, current_qresl_doc_numbers)

            # Append the current list to the ir_eval list
            current_list.append(sys_num)
            current_list.append(query_num)
            current_list.append(round(p_10,3))
            current_list.append(round(r_50,3))
            current_list.append(round(r_precision,3))
            current_list.append(round(ap,3))    
            current_list.append(round(ndcg_10,3))
            current_list.append(round(ndcg_20,3))
            ir_eval.append(current_list)
    
        # ir_eval.append
        current_list = []
        current_list.append(sys_num)
        current_list.append("mean")
        current_list.append(round(statistics.mean([row[2] for row in ir_eval if row[0] == sys_num]),3))
        current_list.append(round(statistics.mean([row[3] for row in ir_eval if row[0] == sys_num]),3))
        current_list.append(round(statistics.mean([row[4] for row in ir_eval if row[0] == sys_num]),3))
        current_list.append(round(statistics.mean([row[5] for row in ir_eval if row[0] == sys_num]),3))
        current_list.append(round(statistics.mean([row[6] for row in ir_eval if row[0] == sys_num]),3))
        current_list.append(round(statistics.mean([row[7] for row in ir_eval if row[0] == sys_num]),3))
        ir_eval.append(current_list)
            
    ir_eval_df = pd.DataFrame(ir_eval, columns=['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20'])
    ir_eval_df['P@10'] = ir_eval_df['P@10'].apply(lambda x: '{:.3f}'.format(x))
    ir_eval_df['R@50'] = ir_eval_df['R@50'].apply(lambda x: '{:.3f}'.format(x))
    ir_eval_df['r-precision'] = ir_eval_df['r-precision'].apply(lambda x: '{:.3f}'.format(x))
    ir_eval_df['AP'] = ir_eval_df['AP'].apply(lambda x: '{:.3f}'.format(x))
    ir_eval_df['nDCG@10'] = ir_eval_df['nDCG@10'].apply(lambda x: '{:.3f}'.format(x))
    ir_eval_df['nDCG@20'] = ir_eval_df['nDCG@20'].apply(lambda x: '{:.3f}'.format(x))
    return ir_eval_df

def calculate_p_10(current_sys_df, current_qresl_doc_numbers):
    """
    Calculate p@10

    Parameters:
    current_sys_df (pd.DataFrame): A dataframe containing the current system results.
    current_qresl_doc_numbers (set): A set containing the current query results.

    Returns:
    float: p@10
    """
    p_10 = 0
    current_sys_doc_10_numbers = set(current_sys_df.head(10)['doc_number'].tolist())
    p_10 = len(current_sys_doc_10_numbers & current_qresl_doc_numbers)
    p_10 = p_10 / 10
    return p_10

def calculate_r_50(current_sys_df, current_qresl_doc_numbers):
    """
    Calculate r@50

    Parameters:
    current_sys_df (pd.DataFrame): A dataframe containing the current system results.
    current_qresl_doc_numbers (set): A set containing the current query results.

    Returns:
    float: r@50
    """
    r_50 = 0
    current_sys_doc_50_numbers = set(current_sys_df.head(50)['doc_number'].tolist())
    r_50 = len(current_sys_doc_50_numbers & current_qresl_doc_numbers)
    r_50 = 0 if len(current_qresl_doc_numbers) == 0 else r_50 / len(current_qresl_doc_numbers)
    return r_50

def calculate_r_precision(current_sys_df, current_qresl_doc_numbers):
    """
    Calculate r-precision

    Parameters:
    current_sys_df (pd.DataFrame): A dataframe containing the current system results.
    current_qresl_doc_numbers (set): A set containing the current query results.

    Returns:
    float: r-precision
    """
    r_precision = 0
    doc_number = len(current_qresl_doc_numbers)
    current_sys_doc_r_numbers = set(current_sys_df.head(doc_number)['doc_number'].tolist())
    r_precision = len(current_sys_doc_r_numbers & current_qresl_doc_numbers)
    r_precision = 0 if doc_number == 0 else r_precision / doc_number
    return r_precision
    
def calculate_ap(current_sys_df, current_qresl_doc_numbers):
    """
    Calculate ap
    
    Parameters:
    current_sys_df (pd.DataFrame): A dataframe containing the current system results.
    current_qresl_doc_numbers (set): A set containing the current query results.

    Returns:
    float: ap
    """
    ap = 0
    ap_doc_number = 0
    current_sys_doc_all_numbers = current_sys_df['doc_number'].tolist()
    for i in range(len(current_sys_doc_all_numbers)):
        if current_sys_doc_all_numbers[i] in current_qresl_doc_numbers:
            ap_doc_number += 1
            ap += (ap_doc_number) / (i + 1)
    
    ap = 0 if ap_doc_number == 0 else ap / ap_doc_number
    return ap

def calculate_ndcg_10(current_sys_df, current_qresl_df ,current_qresl_doc_numbers):
    """
    Calculate ndcg@10
    
    Parameters:
    current_sys_df (pd.DataFrame): A dataframe containing the current system results.
    current_qresl_df (pd.DataFrame): A dataframe containing the current query results.

    Returns:
    float: ndcg@10
    """
    scores = []
    for idx, doc_num in enumerate(current_sys_df.head(10)['doc_number'].tolist()):
        if doc_num in current_qresl_doc_numbers:
            scores.append(current_qresl_df[current_qresl_df['doc_id'] == doc_num]['relevance'].tolist()[0])
        else:
            scores.append(0)

    dcg_10 = calculate_dcg(scores)
    idcg_10 = calculate_dcg(current_qresl_df['relevance'][0:10])
    ndcg_10 = 0 if idcg_10 == 0 else dcg_10 / idcg_10
    return ndcg_10

def calculate_ndcg_20(current_sys_df, current_qresl_df ,current_qresl_doc_numbers):
    """
    Calculate ndcg@20
    
    Parameters:
    current_sys_df (pd.DataFrame): A dataframe containing the current system results.
    current_qresl_df (pd.DataFrame): A dataframe containing the current query results.

    Returns:
    float: ndcg@20
    """
    current_sys_doc_20_numbers = current_sys_df.head(20)['doc_number'].tolist()
    scores = []
    for idx, doc_num in enumerate(current_sys_doc_20_numbers):
        if doc_num in current_qresl_doc_numbers:
            scores.append(current_qresl_df[current_qresl_df['doc_id'] == doc_num]['relevance'].tolist()[0])
        else:
            scores.append(0)

    dcg_20 = calculate_dcg(scores)
    idcg_20 = calculate_dcg(current_qresl_df['relevance'][0:20])
    ndcg_20 = 0 if idcg_20 == 0 else dcg_20 / idcg_20
    return ndcg_20

def calculate_dcg(rels):
    """
    Calculate dcg
    
    Parameters:
    rels (list): A list containing the relevance scores.
    
    Returns:
    float: dcg
    """
    dcg = 0
    for idx, value in enumerate(rels):
        if idx == 0:
            dcg += value
        else:
            dcg += value / np.log2(idx + 1)
    return dcg

# Preprocessing (Tokenization, Stopwords, Stemming)
def regex_tokenisation(text):
    """
    Tokenise a given text using custom regular expression rules.

    This function considers the following rules:
    - Rule 1: Match "ID: \d+"
    - Rule 2: Match words with hyphen e.g., ice-cream
    - Rule 3: Match pure words and words with an apostrophe e.g., can't, i'll

    Parameters:
    text (str): Input string that needs to be tokenised.

    Returns:
    List[str]: A list containing all matched tokens from the input text.
    """

    # Rule 0: Match "ID: \d+"
    rule_1 = r"\bID: \d+\b"

    # Rule 1: Match words with hyphen e.g., ice-cream
    rule_2 = r'\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b'

    # Rule 2: Match pure words and words with an apostrophe e.g., can't, i'll
    rule_3 = r"\b[a-zA-Z]+(?:'[a-zA-Z]+)+\b"

    # Rule 3: Match pure words e.g., hello
    rule_4 = r"\b[a-zA-Z]+\b"

    # Combine rules with OR `|` and use non-capturing group `?:`
    pattern = re.compile(r'(?:{}|{}|{}|{})'.format(rule_1, rule_2, rule_3, rule_4))
    return re.findall(pattern, text)

def get_stop_words(file_path):
    """
    Get stop words from a given file path.

    Before returning the stop words, this function removes all duplicates.

    Parameters:
    file_path (str): File path to the stop words file.

    Returns:
    List[str]: A list containing all stop words from the given file path.
    """

    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()
    return list(set(stop_words))

def normalisation(tokens):
    """
    Normalise a given list of tokens.

    Parameters:
    tokens (List[str]): A list containing all tokens.

    Returns:
    List[str]: A list containing all tokens after normalisation.
    """

    return [token.lower() for token in tokens]

def stopword_removal(tokens, stop_words):
    """
    Remove stop words from a given list of tokens.

    Parameters:
    tokens (List[str]): A list containing all tokens.
    stop_words (List[str]): A list containing all stop words.

    Returns:
    List[str]: A list containing all tokens after stop words removal.
    """
    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    """
    Stem a given list of tokens.

    Parameters:
    tokens (List[str]): A list containing all tokens.

    Returns:
    List[str]: A list containing all tokens after stemming.
    """
    snowball_stemmer = SnowballStemmer('english')
    return [snowball_stemmer.stem(token) for token in tokens]

def write_processed_file(file_path, tokens):
    """
    Write a given tokens to a given file path.

    Parameters:
    file_path (str): File path to the file that needs to be written.
    data (List[str]): A list containing all data that needs to be written.

    
    """
    with open(file_path, 'w') as f:
        f.write(" ".join(tokens))

def pre_process_file_1(data):
    """
    Process a given file.

    This function performs the following steps:
    1. Load data from the given file path.
    2. Tokenise the data.
    3. Normalise the tokens.
    4. Remove stop words from the tokens.
    5. Stem the tokens.
    6. Write the processed tokens to a file.

    Parameters:
    file_path (str): File path to the file that needs to be processed.\
    
    Returns:
    str: File path to the file that contains the processed data.
    """

    # 2.1.1: Load data
    # with open(file_path, 'r') as f:
    #     data = f.read()

    # output_path = file_path.replace(".xml","_processed.txt")

    # 2.1.2: Tokenisation
    tokens = regex_tokenisation(data)

    # 2.1.3: Normalisation
    tokens = normalisation(tokens)

    # 2.1.4: Stop words removal
    # stop_word_path = os.path.join(os.path.dirname(__file__), 'english_stop_words.txt')
    stop_word_path = 'english_stop_words.txt'
    stop_words = get_stop_words(stop_word_path)
    tokens = stopword_removal(tokens, stop_words)

    # 2.1.5: Stemming
    tokens = stemming(tokens)
    return tokens

def generate_doc_dict(lines):
    """
    Generate a dictionary of documents.

    Parameters:
    lines (List[str]): A list containing all lines from the file.

    Returns:
    Dict[str, List[str]]: A dictionary containing all documents.
    """
    document_frequency = defaultdict(int)
    # Read each line from the file
    for tokens in lines:
        # Get the unique tokens in the line
        unique_tokens_in_line = set(tokens)
        for token in unique_tokens_in_line:
            document_frequency[token] += 1
    return document_frequency

def calculate_mutual_information(N11, N01, N10, N00, N):
    """
    Calculate the mutual information given the counts.
    
    Parameters:
    - N11: the number of documents containing the term and belonging to the class.
    - N01: the number of documents not containing the term and belonging to the class.
    - N10: the number of documents containing the term and not belonging to the class.
    - N00: the number of documents not containing the term and not belonging to the class.
    - N: the total number of documents.
    
    Returns:
    - The mutual information score.
    """
    N1_ = N11 + N10
    N_1 = N11 + N01
    N0_ = N01 + N00
    N_0 = N10 + N00
    
    MI = 0
    if N11 > 0:
        MI += (N11 / N) * math.log2((N * N11) / (N1_ * N_1))
    if N01 > 0:
        MI += (N01 / N) * math.log2((N * N01) / (N0_ * N_1))
    if N10 > 0:
        MI += (N10 / N) * math.log2((N * N10) / (N1_ * N_0))
    if N00 > 0:
        MI += (N00 / N) * math.log2((N * N00) / (N0_ * N_0))
    
    return MI

def calculate_chi_squared(N11: int, N01: int, N10: int, N00: int) -> float:
    """
    Calculate the chi-squared score given the counts based on the provided formula.
    
    Parameters:
    - N11: the number of documents containing the term and belonging to the class.
    - N01: the number of documents not containing the term and belonging to the class.
    - N10: the number of documents containing the term and not belonging to the class.
    - N00: the number of documents not containing the term and not belonging to the class.
    
    Returns:
    - The chi-squared score.
    """
    
    # Calculate the chi-squared score
    # Note: The expected frequencies are calculated as (row total * column total) / table total
    # We also need to ensure that we don't divide by zero, so we check if expected values are not zero
    numerator = (N11 + N10 + N01 + N00) * ((N11 * N00 - N10 * N01) ** 2)
    denominator = (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00)
    
    chi_squared = numerator / denominator if denominator != 0 else 0
    
    return chi_squared


def train_lda_and_identify_topics(ot_data, nt_data, quran_data, num_topics=20, passes=15, workers=15, random_state=42):
    """
    Train an LDA model using the given data and identify the main topics for each corpus.

    Parameters:
    ot_data (List[List[str]]): A list containing all OT data.
    nt_data (List[List[str]]): A list containing all NT data.
    quran_data (List[List[str]]): A list containing all Quran data.
    num_topics (int): The number of topics to be identified.
    passes (int): The number of passes to be used for training the LDA model.
    workers (int): The number of workers to be used for training the LDA model.
    random_state (int): The random state to be used for training the LDA model.
    """
    
    # Combine all data
    combined_corpus = ot_data + nt_data + quran_data

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(combined_corpus)
    corpus = [dictionary.doc2bow(text) for text in combined_corpus]

    # Train the LDA model
    lda_model = models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, workers=workers, random_state=random_state)

    # Identify the main topics for each corpus
    ot_topic_distribution = np.zeros(num_topics)
    nt_topic_distribution = np.zeros(num_topics)
    quran_topic_distribution = np.zeros(num_topics)

    for doc in ot_data:
        bow = dictionary.doc2bow(doc)
        for topic_num, topic_prob in lda_model.get_document_topics(bow):
            ot_topic_distribution[topic_num] += topic_prob
    ot_topic_distribution = ot_topic_distribution / len(ot_data)

    for doc in nt_data:
        bow = dictionary.doc2bow(doc)
        for topic_num, topic_prob in lda_model.get_document_topics(bow):
            nt_topic_distribution[topic_num] += topic_prob
    nt_topic_distribution = nt_topic_distribution / len(nt_data)

    for doc in quran_data:
        bow = dictionary.doc2bow(doc)
        for topic_num, topic_prob in lda_model.get_document_topics(bow):
            quran_topic_distribution[topic_num] += topic_prob
    quran_topic_distribution = quran_topic_distribution / len(quran_data)

    ot_main_topics = np.argsort(ot_topic_distribution)[-3:]
    nt_main_topics = np.argsort(nt_topic_distribution)[-3:]
    quran_main_topics = np.argsort(quran_topic_distribution)[-3:]

    print_main_topics(lda_model, ot_main_topics, "OT")
    print_main_topics(lda_model, nt_main_topics, "NT")
    print_main_topics(lda_model, quran_main_topics, "Quran")

def print_main_topics(lda_model, topic_nums, corpus_label):
    """
    Print the main topics for a given corpus.

    Parameters:
    lda_model (gensim.models.ldamodel.LdaModel): The trained LDA model.
    topic_nums (List[int]): A list containing the topic numbers to be printed.
    corpus_label (str): The label of the corpus.
    """
    for topic_num in topic_nums:
        print(f"In {corpus_label}, Topic {topic_num} includes:")
        for word, probability in lda_model.show_topic(topic_num, 10):
            print(f"  {word} - {probability:.3f}")
        print()

def get_ot_mi_quran_mi_chi_and_lda(lines):
    ot_lines = [pre_process_file_1(line.split('\t')[1]) for line in lines if line.split('\t')[0] == "OT"]
    ot_doc_num = len(ot_lines)
    ot_freq_dict = generate_doc_dict(ot_lines)

    nt_lines = [pre_process_file_1(line.split('\t')[1]) for line in lines if line.split('\t')[0] == "NT"]
    nt_doc_num = len(nt_lines)
    nt_freq_dict = generate_doc_dict(nt_lines)

    quran_lines = [pre_process_file_1(line.split('\t')[1]) for line in lines if line.split('\t')[0] == "Quran"]
    quran_doc_num = len(quran_lines)
    quran_freq_dict = generate_doc_dict(quran_lines)

    # Iterate through all unique words
    all_unique_words = set(ot_freq_dict.keys()).union(nt_freq_dict.keys()).union(quran_freq_dict.keys())
    global_freq_dict = defaultdict(lambda: {"ot_word_class":0, "ot_word":0, "ot_class": 0, "ot_no":0, "nt_word_class":0, "nt_word":0, "nt_class": 0, "nt_no":0, "quran_word_class":0, "quran_word":0, "quran_class": 0, "quran_no":0})

    for word in all_unique_words:
        global_freq_dict[word]['ot_word_class'] = ot_freq_dict.get(word, 0)
        global_freq_dict[word]['ot_word'] = nt_freq_dict.get(word, 0) + quran_freq_dict.get(word, 0)
        global_freq_dict[word]['ot_class'] = ot_doc_num - ot_freq_dict.get(word, 0)
        global_freq_dict[word]['ot_no'] = nt_doc_num + quran_doc_num - nt_freq_dict.get(word, 0) - quran_freq_dict.get(word, 0)

        global_freq_dict[word]['nt_word_class'] = nt_freq_dict.get(word, 0)
        global_freq_dict[word]['nt_word'] = ot_freq_dict.get(word, 0) + quran_freq_dict.get(word, 0)
        global_freq_dict[word]['nt_class'] = nt_doc_num - nt_freq_dict.get(word, 0)
        global_freq_dict[word]['nt_no'] = ot_doc_num + quran_doc_num - ot_freq_dict.get(word, 0) - quran_freq_dict.get(word, 0)

        global_freq_dict[word]['quran_word_class'] = quran_freq_dict.get(word, 0)
        global_freq_dict[word]['quran_word'] = ot_freq_dict.get(word, 0) + nt_freq_dict.get(word, 0)
        global_freq_dict[word]['quran_class'] = quran_doc_num - quran_freq_dict.get(word, 0)
        global_freq_dict[word]['quran_no'] = ot_doc_num + nt_doc_num - ot_freq_dict.get(word, 0) - nt_freq_dict.get(word, 0)
    
    total_doc_num = ot_doc_num + nt_doc_num + quran_doc_num
    ot_mi_chi_dict = defaultdict(lambda: {"mi": 0, "chi": 0})
    nt_mi_chi_dict = defaultdict(lambda: {"mi": 0, "chi": 0})
    quran_mi_chi_dict = defaultdict(lambda: {"mi": 0, "chi": 0})

    for token in ot_freq_dict.keys():
        ot_mi_chi_dict[token]['mi'] = calculate_mutual_information(global_freq_dict[token]['ot_word_class'], global_freq_dict[token]['ot_class'], global_freq_dict[token]['ot_word'], global_freq_dict[token]['ot_no'], total_doc_num)
        ot_mi_chi_dict[token]['chi'] = calculate_chi_squared(global_freq_dict[token]['ot_word_class'], global_freq_dict[token]['ot_class'], global_freq_dict[token]['ot_word'], global_freq_dict[token]['ot_no'])

    for token in nt_freq_dict.keys():
        nt_mi_chi_dict[token]['mi'] = calculate_mutual_information(global_freq_dict[token]['nt_word_class'], global_freq_dict[token]['nt_class'], global_freq_dict[token]['nt_word'], global_freq_dict[token]['nt_no'], total_doc_num)
        nt_mi_chi_dict[token]['chi'] = calculate_chi_squared(global_freq_dict[token]['nt_word_class'], global_freq_dict[token]['nt_class'], global_freq_dict[token]['nt_word'], global_freq_dict[token]['nt_no'])

    for token in quran_freq_dict.keys():
        quran_mi_chi_dict[token]['mi'] = calculate_mutual_information(global_freq_dict[token]['quran_word_class'], global_freq_dict[token]['quran_class'], global_freq_dict[token]['quran_word'], global_freq_dict[token]['quran_no'], total_doc_num)
        quran_mi_chi_dict[token]['chi'] = calculate_chi_squared(global_freq_dict[token]['quran_word_class'], global_freq_dict[token]['quran_class'], global_freq_dict[token]['quran_word'], global_freq_dict[token]['quran_no'])
    
    pd.DataFrame(ot_mi_chi_dict).transpose().sort_values(by=['mi'], ascending=False).head(10).reset_index().rename(columns={'index':'token','mi':'score'})[['token', 'score']].to_csv('ot_mi.csv', index=False)
    pd.DataFrame(ot_mi_chi_dict).transpose().sort_values(by=['chi'], ascending=False).head(10).reset_index().rename(columns={'index':'token','chi':'score'})[['token', 'score']].to_csv('ot_chi.csv', index=False)
    pd.DataFrame(nt_mi_chi_dict).transpose().sort_values(by=['mi'], ascending=False).head(10).reset_index().rename(columns={'index':'token','mi':'score'})[['token', 'score']].to_csv('nt_mi.csv', index=False)
    pd.DataFrame(nt_mi_chi_dict).transpose().sort_values(by=['chi'], ascending=False).head(10).reset_index().rename(columns={'index':'token','chi':'score'})[['token', 'score']].to_csv('nt_chi.csv', index=False)
    pd.DataFrame(quran_mi_chi_dict).transpose().sort_values(by=['mi'], ascending=False).head(10).reset_index().rename(columns={'index':'token','mi':'score'})[['token', 'score']].to_csv('quran_mi.csv', index=False)
    pd.DataFrame(quran_mi_chi_dict).transpose().sort_values(by=['chi'], ascending=False).head(10).reset_index().rename(columns={'index':'token','chi':'score'})[['token', 'score']].to_csv('quran_chi.csv', index=False)

    train_lda_and_identify_topics(ot_lines, nt_lines, quran_lines, num_topics=20, random_state=42)

def baseline_tokenisation(text):
    """
    Tokenise a given text using custom regular expression rules.

    This function considers the following rules:
    - Rule 1: Match words with hyphen e.g., ice-cream
    - Rule 2: Match pure words and words with an apostrophe e.g., can't, i'll

    Parameters:
    text (str): Input string that needs to be tokenised.

    Returns:
    List[str]: A list containing all matched tokens from the input text.
    """

    # Rule 1: Match words with hyphen e.g., ice-cream
    rule_1 = r'\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b'

    # Rule 2: Match pure words and words with an apostrophe e.g., can't, i'll
    rule_2 = r"\b[a-zA-Z]+(?:'[a-zA-Z]+)+\b"

    # Rule 3: Match pure words e.g., hello
    rule_3 = r"\b[a-zA-Z]+\b"

    # Combine rules with OR `|` and use non-capturing group `?:`
    pattern = re.compile(r'(?:{}|{}|{})'.format(rule_1, rule_2, rule_3))
    return re.findall(pattern, text)

def remove_url(text):
    """
    Remove URL from a given text.

    Parameters:
    text (str): Input string that needs to be processed.

    Returns:
    str: The processed string.
    """
    return re.sub(r"http\S+", "", text)

def create_vocabulary(tweets):
    vocabulary = {}
    for tweet in tweets:
        for word in tweet:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)
    return vocabulary

def corpus_to_bow_dok(tweets, vocabulary):
    # Create a DOK matrix
    bow_matrix = dok_matrix((len(tweets), len(vocabulary)), dtype=np.int32)
    
    for doc_idx, document in enumerate(tweets):
        for word in document:
            if word in vocabulary:
                word_idx = vocabulary[word]
                bow_matrix[doc_idx, word_idx] += 1
    return bow_matrix

def calculate_tp_fp_fn(y_true, y_pred, class_label):
    TP = sum((y_pred == class_label) & (y_true == class_label))
    FP = sum((y_pred == class_label) & (y_true != class_label))
    FN = sum((y_pred != class_label) & (y_true == class_label))
    return TP, FP, FN

def calculate_precision_recall_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1

def evaluate_model(y_true, y_pred, class_labels):
    precisions, recalls, f1_scores = [], [], []
    for class_label in class_labels:
        TP, FP, FN = calculate_tp_fp_fn(y_true, y_pred, class_label)
        precision, recall, f1 = calculate_precision_recall_f1(TP, FP, FN)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    p_macro = sum(precisions) / len(precisions)
    r_macro = sum(recalls) / len(recalls)
    f_macro = sum(f1_scores) / len(f1_scores)
    return precisions, recalls, f1_scores, p_macro, r_macro, f_macro

def base_line():
    train_data = pd.read_csv('train.txt', sep='\t')
    test_data = pd.read_csv('ttds_2023_cw2_test.txt', sep='\t')
    train_data['tweet'] = train_data['tweet'].apply(lambda x: remove_url(x))
    train_data['tweet'] = train_data['tweet'].apply(lambda x: baseline_tokenisation(x))
    test_data['tweet'] = test_data['tweet'].apply(lambda x: remove_url(x))
    test_data['tweet'] = test_data['tweet'].apply(lambda x: baseline_tokenisation(x))
    
    # Split the data into train and dev
    X_train, X_dev, y_train, y_dev = train_test_split(train_data['tweet'], train_data['sentiment'], test_size=0.1, random_state=42)
    # Create a vocabulary
    train_tweets = [tweet for tweet in X_train]
    vocabulary = create_vocabulary(train_tweets)
    bow_train_tweets = corpus_to_bow_dok(train_tweets, vocabulary)

    dev_tweets = [tweet for tweet in X_dev]
    bow_dev_tweets = corpus_to_bow_dok(dev_tweets, vocabulary)

    test_tweets = [tweet for tweet in test_data['tweet']]
    bow_test_tweets = corpus_to_bow_dok(test_tweets, vocabulary)

    # Train the model
    svm = SVC(C=1000)
    svm.fit(bow_train_tweets, y_train)

    # Make predictions
    train_predictions = svm.predict(bow_train_tweets)
    dev_predictions = svm.predict(bow_dev_tweets)
    test_predictions = svm.predict(bow_test_tweets)

    # Evaluate the model
    class_labels = ['positive', 'negative', 'neutral']
    train_precisions, train_recalls, train_f1s, train_p_macro, train_r_macro, train_f_macro = evaluate_model(y_train, train_predictions, class_labels)
    dev_precisions, dev_recalls, dev_f1s, dev_p_macro, dev_r_macro, dev_f_macro = evaluate_model(y_dev, dev_predictions, class_labels)
    test_precisions, test_recalls, test_f1s, test_p_macro, test_r_macro, test_f_macro = evaluate_model(test_data['sentiment'], test_predictions, class_labels)

    # Print the results
    print("system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro")
    print(f"baseline,train,{train_precisions[0]},{train_recalls[0]},{train_f1s[0]},{train_precisions[1]},{train_recalls[1]},{train_f1s[1]},{train_precisions[2]},{train_recalls[2]},{train_f1s[2]},{train_p_macro},{train_r_macro},{train_f_macro}")
    print(f"baseline,dev,{dev_precisions[0]},{dev_recalls[0]},{dev_f1s[0]},{dev_precisions[1]},{dev_recalls[1]},{dev_f1s[1]},{dev_precisions[2]},{dev_recalls[2]},{dev_f1s[2]},{dev_p_macro},{dev_r_macro},{dev_f_macro}")
    print(f"baseline,test,{test_precisions[0]},{test_recalls[0]},{test_f1s[0]},{test_precisions[1]},{test_recalls[1]},{test_f1s[1]},{test_precisions[2]},{test_recalls[2]},{test_f1s[2]},{test_p_macro},{test_r_macro},{test_f_macro}")

    # Save the results to a dataframe
    results = {
        'system': ['baseline', 'baseline', 'baseline'],
        'split': ['train', 'dev', 'test'],
        'p-pos': [train_precisions[0], dev_precisions[0], test_precisions[0]],
        'r-pos': [train_recalls[0], dev_recalls[0], test_recalls[0]],
        'f-pos': [train_f1s[0], dev_f1s[0], test_f1s[0]],
        'p-neg': [train_precisions[1], dev_precisions[1], test_precisions[1]],
        'r-neg': [train_recalls[1], dev_recalls[1], test_recalls[1]],
        'f-neg': [train_f1s[1], dev_f1s[1], test_f1s[1]],
        'p-neu': [train_precisions[2], dev_precisions[2], test_precisions[2]],
        'r-neu': [train_recalls[2], dev_recalls[2], test_recalls[2]],
        'f-neu': [train_f1s[2], dev_f1s[2], test_f1s[2]],
        'p-macro': [train_p_macro, dev_p_macro, test_p_macro],
        'r-macro': [train_r_macro, dev_r_macro, test_r_macro],
        'f-macro': [train_f_macro, dev_f_macro, test_f_macro]
    }
    print(classification_report(test_data['sentiment'], test_predictions))
    df = pd.DataFrame(results)
    return df

def improved_model():
    train_data = pd.read_csv('train.txt', sep='\t')
    test_data = pd.read_csv('ttds_2023_cw2_test.txt', sep='\t')
    train_data['tweet'] = train_data['tweet'].apply(lambda x: remove_url(x))
    train_data['tweet'] = train_data['tweet'].apply(lambda x: ' '.join(pre_process_file_1(x)))
    test_data['tweet'] = test_data['tweet'].apply(lambda x: remove_url(x))
    test_data['tweet'] = test_data['tweet'].apply(lambda x: ' '.join(pre_process_file_1(x)))

    X_train, X_dev, y_train, y_dev = train_test_split(train_data['tweet'], train_data['sentiment'], test_size=0.1, random_state=42)
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(X_train)
    tfidf_dev = vectorizer.transform(X_dev)
    tfidf_test = vectorizer.transform(test_data['tweet'])

    # Train the model
    svm = SVC(C=1000)
    svm.fit(tfidf_train, y_train)

    # Make predictions
    train_predictions = svm.predict(tfidf_train)
    dev_predictions = svm.predict(tfidf_dev)
    test_predictions = svm.predict(tfidf_test)

    # Evaluate the model
    class_labels = ['positive', 'negative', 'neutral']
    train_precisions, train_recalls, train_f1s, train_p_macro, train_r_macro, train_f_macro = evaluate_model(y_train, train_predictions, class_labels)
    dev_precisions, dev_recalls, dev_f1s, dev_p_macro, dev_r_macro, dev_f_macro = evaluate_model(y_dev, dev_predictions, class_labels)
    test_precisions, test_recalls, test_f1s, test_p_macro, test_r_macro, test_f_macro = evaluate_model(test_data['sentiment'], test_predictions, class_labels)

    results = {
        'system': ['improved', 'improved', 'improved'],
        'split': ['train', 'dev', 'test'],
        'p-pos': [train_precisions[0], dev_precisions[0], test_precisions[0]],
        'r-pos': [train_recalls[0], dev_recalls[0], test_recalls[0]],
        'f-pos': [train_f1s[0], dev_f1s[0], test_f1s[0]],
        'p-neg': [train_precisions[1], dev_precisions[1], test_precisions[1]],
        'r-neg': [train_recalls[1], dev_recalls[1], test_recalls[1]],
        'f-neg': [train_f1s[1], dev_f1s[1], test_f1s[1]],
        'p-neu': [train_precisions[2], dev_precisions[2], test_precisions[2]],
        'r-neu': [train_recalls[2], dev_recalls[2], test_recalls[2]],
        'f-neu': [train_f1s[2], dev_f1s[2], test_f1s[2]],
        'p-macro': [train_p_macro, dev_p_macro, test_p_macro],
        'r-macro': [train_r_macro, dev_r_macro, test_r_macro],
        'f-macro': [train_f_macro, dev_f_macro, test_f_macro]
    }
    print(classification_report(test_data['sentiment'], test_predictions))
    df = pd.DataFrame(results)
    return df

if __name__ == '__main__':
    # Task 1 - IR Evaluation
    sys_results = pd.read_csv('systemresults.csv')
    qresl = pd.read_csv('qrels.csv')
    ir_eval_df = ir_eval(sys_results, qresl)
    ir_eval_df.to_csv('ir_eval.csv', index=False)
    print("Task 1 - IR Evaluation: Done")

    # Task2 - Text Analysis
    with open('train_and_dev.tsv', 'r') as f:
        train_and_dev_data = f.read()
        lines = train_and_dev_data.split('\n')[:-1]

    get_ot_mi_quran_mi_chi_and_lda(lines)
    print("Task 2 - Text Analysis: Done")

    # Task3 - Text Classification
    train_data = pd.read_csv('train.txt', sep='\t')
    test_data = pd.read_csv('ttds_2023_cw2_test.txt', sep='\t')

    baseline_df = base_line()
    print("Task 3 - Baseline: Done")
    print()
    improved_df = improved_model()
    print("Task 3 - Improved: Done")
    # Combine the results
    df = pd.concat([baseline_df, improved_df])
    csv_file_path = 'classification_report.csv'
    df.to_csv(csv_file_path, index=False)




    