import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier

# Load datasets
train_df = pd.read_excel('./Data/q/processed_train.xlsx')
test_df = pd.read_excel('./Data/q/processed_test.xlsx')
valid_df = pd.read_excel('./Data/q/processed_valid.xlsx')

# Extract Year (if necessary)
# Ensure 'Year' column exists or extract from ID if needed
# e.g., train_df['Year'] = train_df.ID.map(extractYearFromId).astype(np.int64)

# Extract non-null CVSS2 data
# train_df = train_df[train_df['CVSS2_Avail'].notnull()]
# test_df = test_df[test_df['CVSS2_Avail'].notnull()]
# valid_df = valid_df[valid_df['CVSS2_Avail'].notnull()]

# Define features and labels
X_train = train_df['processed_description'].values
X_test = test_df['processed_description'].values
X_valid = valid_df['processed_description'].values

# Define labels
y_train = train_df['Base Severity'].values
y_test = test_df['Base Severity'].values
y_valid = valid_df['Base Severity'].values

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_valid = label_encoder.transform(y_valid)

# Define feature extraction
def extract_features(config, start_word_ngram, end_word_ngram):
    if config == 1:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=False, min_df=0.001,
                               norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')
    elif config == 2:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=True, min_df=0.001,
                               norm='l2', token_pattern=r'\S*[A-Za-z]\S+')
    elif config < 6:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=False,
                               min_df=0.001, norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')
    return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=True,
                           min_df=0.001, norm='l2', token_pattern=r'\S*[A-Za-z]\S+')

# Build classifiers
def build_classifiers(config):
    clfs = {
        'NB': MultinomialNB(),
        'SVM': OneVsRestClassifier(LinearSVC(random_state=42, C=0.1, max_iter=1000), n_jobs=-1)
    }
    if config == 2 or config == 6 or config == 7 or config == 8:
        clfs['LR'] = LogisticRegression(C=10, multi_class='ovr', n_jobs=-1, solver='lbfgs', max_iter=1000, random_state=42)
    else:
        clfs['LR'] = LogisticRegression(C=0.1, multi_class='ovr', n_jobs=-1, solver='lbfgs', max_iter=1000, random_state=42)
    clfs['RF'] = RandomForestClassifier(n_estimators=100, max_depth=None, max_leaf_nodes=None, random_state=42, n_jobs=-1)
    clfs['XGB'] = XGBClassifier(objective='multiclass', max_depth=0, max_leaves=100, grow_policy='lossguide', n_jobs=-1, random_state=42, tree_method='hist')
    clfs['LGBM'] = LGBMClassifier(num_leaves=100, max_depth=-1, objective='multiclass', n_jobs=-1, random_state=42)
    return clfs

# Feature model
def feature_model(X_train, X_test, y_test, config, start_word_ngram, end_word_ngram):
    vectorizer = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram)
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    # Remove rows with all zero values
    test_df = pd.DataFrame(X_test_transformed.todense())
    results = test_df.apply(lambda x: x.value_counts().get(0.0, 0), axis=1)
    non_zero_indices = np.where(results < len(test_df.columns))[0]
    X_train_transformed = X_train_transformed.astype(np.float64)
    X_test_transformed = X_test_transformed.astype(np.float64)
    return X_train_transformed, X_test_transformed[non_zero_indices], y_test[non_zero_indices]

# Evaluate model
def evaluate(clf, X_train_transformed, X_test_transformed, y_train, y_test):
    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)
    precisionma, recallma, f1ma, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    return acc, precisionma, recallma, f1ma, mcc

# Validate data (no cross-validation, use provided datasets)
def validate_data(clf, config, start_word_ngram, end_word_ngram):
    t_start = time.perf_counter()
    X_train_transformed, X_valid_transformed, y_valid_filtered = feature_model(X_train, X_valid, y_valid, config, start_word_ngram, end_word_ngram)
    results = evaluate(clf, X_train_transformed, X_valid_transformed, y_train, y_valid_filtered)
    val_time = time.perf_counter() - t_start
    return "{:.5f}".format(results[0]) + "\t" + "{:.5f}".format(results[1]) + "\t" + "{:.5f}".format(results[2]) + "\t" + "{:.5f}".format(results[3]) + "\t" + "{:.5f}".format(results[4]) + "\t" + "{:.5f}".format(val_time)

# Labels to evaluate
labels = ['Base Severity']

# Configurations
configs = list(range(1, 9))

# Results file
import datetime
result_file = datetime.datetime.now().strftime('val_with_time' + '%Y_%m_%d_%H_%M_%S') + '.txt'

with open(result_file, 'w') as fout:
    for config in configs:
        print("Current config:", config)
        fout.write("Current config:" + str(config) + "\n")
        start_word_ngram = 1
        end_word_ngram = 1
        if config == 1:
            print("Bag-of-word without tf-idf")
            fout.write("Bag-of-word without tf-idf\n")
        elif config == 2:
            print("Bag-of-word with tf-idf")
            fout.write("Bag-of-word with tf-idf\n")
        elif config <= 5:
            print("N-gram without tf-idf")
            fout.write("N-gram without tf-idf\n")
            if config == 3:
                end_word_ngram = 2
            elif config == 4:
                end_word_ngram = 3
            elif config == 5:
                end_word_ngram = 4
        else:
            print("N-gram with tf-idf")
            fout.write("N-gram with tf-idf\n")
            if config == 6:
                end_word_ngram = 2
            elif config == 7:
                end_word_ngram = 3
            elif config == 8:
                end_word_ngram = 4
        clfs = build_classifiers(config=config)
        for label in labels:
            print("Current output:", label, "\n")
            fout.write("Current output:" + label + "\n")
            print("Classifier\tAccuracy\tprecision\trecall\tf1\tmcc\tVal Time\n")
            fout.write("Classifier\tAccuracy\tprecision\trecall\tf1\tmcc\tVal Time\n\n")
            for clf_name, clf in clfs.items():
                print(clf_name + "\t", end='')
                fout.write(clf_name + "\t")
                val_res = validate_data(clf, config, start_word_ngram, end_word_ngram)
                print(val_res)
                fout.write(val_res + "\n")
            print("------------------------------------------------\n")
            fout.write("------------------------------------------------\n\n")
        print("##############################################\n")
        fout.write("##############################################\n\n")
