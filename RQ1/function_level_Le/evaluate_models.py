import pandas as pd
import numpy as np
import pickle
import math
import datetime
import time
import sys
import os
import itertools
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, \
	matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from helpers import *
from helpers.tokenizer import *
from scipy.sparse import hstack, vstack, coo_matrix
from gensim.models import FastText, Word2Vec
# from imblearn.over_sampling import RandomOverSampler, SMOTE
sys.path.append(str((Path(__file__).parent.parent.parent)))
import gcn
import csv


import warnings
warnings.filterwarnings("ignore")
# Local imports
from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')
os.environ['PYTHONHASHSEED'] = '42'





########################################################################################

def most_common(l):
	return Counter(l).most_common(1)[0][0]


def sen_to_vec(words, model):
	# words = sen.split()

	sen_vec = np.array([0.0] * 300)
	cnt = 0

	for w in words:
		try:
			sen_vec = sen_vec + model.wv[w]
			cnt += 1
		except:
			pass

	if cnt == 0:
		return np.random.rand(300)

	return sen_vec / (cnt * 1.0)


def infer_features_sen(sentences, model):
	feature_vectors = []

	for sentence in sentences:
		feature_vectors.append(sen_to_vec(sentence, model))

	return np.asarray(feature_vectors)


def infer_features_d2v(sentences, model):
	feature_vectors = []

	for sentence in sentences:
		feature_vectors.append(model.infer_vector(sentence))

	return np.asarray(feature_vectors)


def generate_features(x_train, x_val, x_test, feature='bow'):
	print(type(x_train))
	t = x_train.tolist()
	print(len(t))
	# print(t)
	feature = feature.strip()
	if feature == 'bow':
		code_token_pattern = gen_tok_pattern()
		# vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		x_train_transformed = vectorizer.fit_transform(x_train)
		x_val_transformed = vectorizer.transform(x_val)
		x_test_transformed = vectorizer.transform(x_test)

		print(len(vectorizer.vocabulary_))

	elif feature == 'subwords':
		code_token_pattern = gen_tok_pattern()
		
		# vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		vectorizer.fit(x_train)
		word_vocab = vectorizer.vocabulary_

		start_gram = 3
		end_gram = 6

		use_idf = False
		norm = None

		tf_idf = TfidfVectorizer(ngram_range=(start_gram, end_gram), use_idf=use_idf, min_df=10,
								 analyzer='char', norm=norm)

		tf_idf.fit(x_train)

		char_vocabs = tf_idf.vocabulary_

		slt_char_vocabs = []
		for w in char_vocabs.keys():
			toks = w.split()
			if len(toks) == 1 and len(toks[0]) > 1:
				slt_char_vocabs.append(w.strip())

		slt_char_vocabs = set(slt_char_vocabs)

		# print(slt_char_vocabs)
		print(len(slt_char_vocabs))

		word_vocab = set(word_vocab) - slt_char_vocabs

		tf_idf_char = TfidfVectorizer(stop_words=['aka'], ngram_range=(start_gram - 1, end_gram), use_idf=use_idf,
									  min_df=0.0, analyzer='char', norm=norm, vocabulary=slt_char_vocabs)
		x_train_char = tf_idf_char.fit_transform(x_train)
		x_val_char = tf_idf_char.transform(x_val)
		x_test_char = tf_idf_char.transform(x_test)

		# vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern,
		# 							  vocabulary=word_vocab)
		vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern,
									  vocabulary=word_vocab)
		x_train_word = vectorizer.fit_transform(x_train)
		x_val_word = vectorizer.transform(x_val)
		x_test_word = vectorizer.transform(x_test)

		x_train_transformed = hstack([x_train_word, x_train_char])
		x_val_transformed = hstack([x_val_word, x_val_char])
		x_test_transformed = hstack([x_test_word, x_test_char])

	elif feature == 'word2vec':
		model = Word2Vec(vector_size=300, window=5, min_count=2, workers=1, seed=42, sg=1)
		code_token_pattern = gen_tok_pattern()
		# vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		analyzer = vectorizer.build_analyzer()

		train_text = []
		for i in range(len(x_train)):
			train_text.append(analyzer(x_train[i]))

		val_text = []
		for i in range(len(x_val)):
			val_text.append(analyzer(x_val[i]))

		test_text = []
		for i in range(len(x_test)):
			test_text.append(analyzer(x_test[i]))

		# print(train_text)

		model.build_vocab(corpus_iterable=train_text)
		model.train(corpus_iterable=train_text, total_examples=len(train_text), epochs=10)
		print(len(model.wv))

		x_train_transformed = infer_features_sen(train_text, model)
		x_val_transformed = infer_features_sen(val_text, model)
		x_test_transformed = infer_features_sen(test_text, model)

	elif feature == 'fasttext':
		model = FastText(vector_size=300, window=5, min_count=2, workers=1, seed=42, sg=1, min_n=3, max_n=6)
		code_token_pattern = gen_tok_pattern()
		# vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		vectorizer = extract_features(start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
		analyzer = vectorizer.build_analyzer()

		train_text = []
		for i in range(len(x_train)):
			train_text.append(analyzer(x_train[i]))

		val_text = []
		for i in range(len(x_val)):
			val_text.append(analyzer(x_val[i]))

		test_text = []
		for i in range(len(x_test)):
			test_text.append(analyzer(x_test[i]))

		model.build_vocab(corpus_iterable=train_text)
		model.train(corpus_iterable=train_text, total_examples=len(train_text), epochs=10)

		print(len(model.wv))

		x_train_transformed = infer_features_sen(train_text, model)
		x_val_transformed = infer_features_sen(val_text, model)
		x_test_transformed = infer_features_sen(test_text, model)

	elif feature == 'codebert':
		# np.asarray([float(i) for i in r.codebert[1:-1].split(",")]
		# x_train = x_train.tolist()
		for i in range(len(x_train)):
			x_train[i] = [float(i) for i in x_train[i][1:-1].split(",")]
		for i in range(len(x_val)):
			x_val[i] = [float(i) for i in x_val[i][1:-1].split(",")]
		for i in range(len(x_test)):
			x_test[i] = [float(i) for i in x_test[i][1:-1].split(",")]
		x_train_transformed = np.asarray(x_train.tolist())
		x_val_transformed = np.asarray(x_val.tolist())
		x_test_transformed = np.asarray(x_test.tolist())
		# print(x_train_transformed[0][0])
		# print(x_train_transformed.shape)
		# print(x_val_transformed.shape)
		# print(x_test_transformed.shape)

	x_train_transformed = x_train_transformed.astype(np.float64)
	x_val_transformed = x_val_transformed.astype(np.float64)
	x_test_transformed = x_test_transformed.astype(np.float64)

	return x_train_transformed, x_val_transformed, x_test_transformed


#########################################################################################


def get_classifier(alg, multiclass, num_class, *parameters):
	# Logistic Regression.

	workers = 16

	if alg == 'lr':
		if multiclass:
			problem_type = 'multinomial'
		else:
			problem_type = 'ovr'
		return LogisticRegression(C=float(parameters[0]), multi_class=problem_type, n_jobs=workers, solver='lbfgs',
								  tol=0.001, max_iter=10000, random_state=42)
	# Support Vector Machine
	elif alg == 'svm':
		# return SVC(random_state=42, C=float(parameters[0]), kernel='rbf', max_iter=-1, probability=True)
		return OneVsRestClassifier(LinearSVC(C=float(parameters[0]), random_state=42, max_iter=10000), n_jobs=workers)
	# K-Nearest Neighbours
	elif alg == 'knn':
		return KNeighborsClassifier(n_neighbors=int(parameters[0]), weights=parameters[1], p=int(parameters[2]),
									n_jobs=workers)
	# Random Forest
	elif alg == 'rf':
		return RandomForestClassifier(n_estimators=int(parameters[0]), max_depth=None,
									  max_leaf_nodes=int(parameters[1]), random_state=42, n_jobs=workers)
	# Extreme Gradient Boosting
	elif alg == 'xgb':
		if multiclass:
			problem_type = 'multi:softmax'
			return XGBClassifier(objective=problem_type, max_depth=0, n_estimators=int(parameters[0]),
								 max_leaves=int(parameters[1]), grow_policy='lossguide', n_jobs=workers,
								 random_state=42, tree_method='hist', num_class=num_class)

		else:
			problem_type = 'binary:logistic'
			return XGBClassifier(objective=problem_type, max_depth=0, n_estimators=int(parameters[0]),
								 max_leaves=int(parameters[1]), grow_policy='lossguide', n_jobs=workers,
								 random_state=42, tree_method='hist')

	# Light Gradient Boosting Machine
	elif alg == 'lgbm':
		if multiclass:
			problem_type = 'multiclass'
		else:
			problem_type = 'binary'
		return LGBMClassifier(n_estimators=int(parameters[0]), num_leaves=int(parameters[1]), max_depth=-1,
							  objective=problem_type, n_jobs=workers, random_state=42)


#########################################################################################


def extract_results(y_true, y_pred):
	# Evaluate
	if len(np.unique(y_true)) == 1:
		precision = accuracy_score(y_true, y_pred)
		recall = precision
		f1 = precision
	#because deepcva and our model are multi-task, and predict all result simultaneously, so we all choose macro for every cvss
	#For fairness, we annotation it, even if it is single-task model
	# elif len(np.unique(np.r_[y_true, y_pred])) == 2:

	# 	precisions = []
	# 	recalls = []

	# 	for label in np.unique(np.r_[y_true, y_pred]):
	# 		precisions.append(precision_score(y_true, y_pred, average='binary', pos_label=label))
	# 		recalls.append(recall_score(y_true, y_pred, average='binary', pos_label=label))

	# 	precision = np.mean(precisions)
	# 	recall = np.mean(recalls)
	# 	f1 = (2 * precision * recall) / (precision + recall)

	else:
		precision = precision_score(y_true, y_pred, average='macro')
		recall = recall_score(y_true, y_pred, average='macro')
		f1 = f1_score(y_true, y_pred, average='macro')

	gmean = math.sqrt(recall * precision)

	output = f"{round(accuracy_score(y_true, y_pred),3)},{round(precision,3)},{round(recall,3)}," + \
			 f"{round(gmean,3)},{round(f1,3)},{round(matthews_corrcoef(y_true, y_pred),3)}"

	return output


#########################################################################################


# Train, Evaluate and Save a Classifier
def evaluate(clf, x_train, y_train, x_val, y_val, x_test, y_test, clf_settings, outpath, write=True):
	# Open the results file
	if not os.path.exists(outpath):
		outfile = open(outpath, 'w')

		outfile.write("problem,granularity,scope,feature,classifier,parameters,"
					  "val_acc,val_prec,val_rec,val_gmean,val_f1,val_mcc,train_time,val_time,"
					  "test_acc,test_prec,test_rec,test_gmean,test_f1,test_mcc,test_time\n")
	else:
		outfile = open(outpath, 'a')
	# Train
	t_start = time.time()
	clf.fit(x_train, y_train)
	train_time = time.time() - t_start
	# Predict
	p_val_start = time.time()
	y_val_pred = clf.predict(x_val)
	val_time = time.time() - p_val_start

	# Predict
	p_test_start = time.time()
	y_test_pred = clf.predict(x_test)
	test_time = time.time() - p_test_start

	output = f"{clf_settings},"

	output += extract_results(y_val, y_val_pred) + f",{round(train_time,3)},{round(val_time,3)},"
	output += extract_results(y_test, y_test_pred) + f",{round(test_time,3)}\n"

	# Save results
	if write:
		outfile.write(output)


#########################################################################################

def compute_average_results(result_file):
	avg_df = pd.read_csv(result_file)

	avg_df = avg_df.groupby(
		['problem', 'granularity', 'scope', 'feature', 'classifier', 'parameters']).mean().reset_index()

	return avg_df


def find_best_results(avg_df, group_cols=['problem', 'granularity', 'scope']):
	return avg_df.loc[avg_df.groupby(group_cols)['val_mcc'].idxmax()].reset_index(drop=True)


# Hyper-parameters.
regularization_lr = ['0.01', '0.1', '1', '10', '100']  # Regularization Coefficient for LR
regularization_svm = ['0.01', '0.1', '1', '10', '100']  # Regularization Coefficient for SVM
neighbours = ['5', '11', '31', '51']  # Number of Neighbours for KNN
weights = ['uniform', 'distance']  # Distance Weight for KNN
norms = ['1', '2']  # Distance Norm for KNN
estimators = ['100', '200', '300', '400', '500']  # Number of estimators for RF, XGB, LGBM
leaf_nodes = ['100', '200', '300']  # Number of leaf nodes for RF, XGB, LGBM

models = ['svm', 'lr', 'knn', 'rf', 'xgb', 'lgbm']
models = ['rf', 'lgbm'] #choose LGBM+BoST and RF+BoST
def main(granularity, scope, cvss_col, feature):
	print('-------------------------------')
	print(granularity, scope, cvss_col, feature)
	print("-------------------------------")
	print('Current time:', datetime.datetime.now())

	start_time = time.time()
	result_folder = gcn.get_dir(gcn.function_Le_dir()/ "ml_results_single")

	classes = cvss_col

	result_file = result_folder  / f'results_{granularity}_{scope}_{cvss_col}_{feature}.csv'

	if os.path.exists(result_file):
		os.remove(result_file)

	key_col = 'key'
	# fold_col = 'fold'

	data_folder = gcn.data_dir() / "function_level_Le_Data"

	# df_map = pd.read_csv(data_folder / (granularity + '_map.csv'))
	# df_map[fold_col] = df_map[fold_col].astype(int)
	# df_map[key_col] = df_map[key_col].astype(str)

	if feature == 'codebert':
		data_col = 'codebert'
		filename = data_folder /( granularity + '_' + scope + '_codebert.csv')
	else:
		data_col = 'code'
		filename = data_folder / (granularity + '_' + scope + '.csv')

	df = pd.read_csv(filename)
	cvss3_severity_dict = {"LOW":0,"MEDIUM":1,"HIGH":2,"CRITICAL":3}

	df["cvss3_severity"] = df[["cvss3_severity"]].apply(
	lambda r: cvss3_severity_dict[r.cvss3_severity], axis=1
	)
	# AV_dict = {"N":0,"A":1,"L":2,"P":3}
	# df["AV"] = df[["AV"]].apply(
	# lambda r: AV_dict[r.AV], axis=1
	# )
	# AC_dict = {"L":0,"H":1}
	# df["AC"] = df[["AC"]].apply(
	# 	lambda r: AC_dict[r.AC], axis=1
	# )
	# PR_dict = {"N":0,"L":1,"H":2}
	# df["PR"] = df[["PR"]].apply(
	# 	lambda r: PR_dict[r.PR], axis=1
	# )
	# UI_dict = {"N":0,"R":1}
	# df["UI"] = df[["UI"]].apply(
	# 	lambda r: UI_dict[r.UI], axis=1
	# )
	# S_dict = {"U":0,"C":1}
	# df["S"] = df[["S"]].apply(
	# 	lambda r: S_dict[r.S], axis=1
	# )
	# C_dict = {"N":0,"L":1,"H":2}
	# df["C"] = df[["C"]].apply(
	# 	lambda r: C_dict[r.C], axis=1
	# )
	# I_dict = {"N": 0, "L": 1, "H": 2}
	# df["I"] = df[["I"]].apply(
	# 	lambda r: I_dict[r.I], axis=1
	# )
	# A_dict = {"N": 0, "L": 1, "H": 2}
	# df["A"] = df[["A"]].apply(
	# 	lambda r: A_dict[r.A], axis=1
	# )

	df["cvss3_severity"] = df["cvss3_severity"].astype("int")
	# df["AV"] = df["AV"].astype("int")
	# df["AC"] = df["AC"].astype("int")
	# df["PR"] = df["PR"].astype("int")
	# df["UI"] = df["UI"].astype("int")
	# df["S"] = df["S"].astype("int")
	# df["C"] = df["C"].astype("int")
	# df["I"] = df["I"].astype("int")
	# df["A"] = df["A"].astype("int")

	if feature != 'codebert':
		df = df.applymap(str)
	# else:
	# 	df[key_col] = df[key_col].astype(str)

	# folds = np.sort(df_map['fold'].unique())

	# print(granularity, scope, cvss_col, feature)

	# for index in range(len(folds)):

	# 	fold = folds[index]

	# 	val_index = (index + 1) % len(folds)
	# 	test_index = (index + 2) % len(folds)

	# 	train_indices = [i for i in range(len(folds)) if i != val_index and i != test_index]

	# 	print(train_indices)
	# 	print(val_index)
	# 	print(test_index)

	# 	train_entries = df_map[df_map[fold_col].isin(train_indices)][key_col].values
	# 	val_entries = df_map[df_map[fold_col] == val_index][key_col].values
	# 	test_entries = df_map[df_map[fold_col] == test_index][key_col].values

	# 	train_df = df[df[key_col].isin(train_entries)]
	# 	val_df = df[df[key_col].isin(val_entries)]
	# 	test_df = df[df[key_col].isin(test_entries)]

	# 	x_train = train_df[data_col].values
	# 	x_val = val_df[data_col].values
	# 	x_test = test_df[data_col].values

	# 	y_train = train_df[cvss_col].values
	# 	print(type(y_train))
	# 	print(y_train)
	# 	y_train = y_train.astype("int")
		
	# 	y_val = val_df[cvss_col].values
	# 	y_val = y_val.astype("int")
	# 	y_test = test_df[cvss_col].values
	# 	y_test = y_test.astype("int")
	# 	print(folds[index], len(x_train), Counter(y_train))

	# 	x_train, x_val, x_test = generate_features(x_train, x_val, x_test, feature)

	# 	for model_name in models:

	# 		param_set = []

	# 		if model_name == 'lr':
	# 			param_set = list(itertools.product(*[regularization_lr]))
	# 		elif model_name == 'svm':
	# 			param_set = list(itertools.product(*[regularization_svm]))
	# 		elif model_name == 'knn':
	# 			param_set = list(itertools.product(*[neighbours, weights, norms]))
	# 		elif model_name == 'rf' or model_name == 'xgb' or model_name == 'lgbm':
	# 			param_set = list(itertools.product(*[estimators, leaf_nodes]))
	# 		# Run for each parameter configuration

	# 		for parameters in param_set:
	# 			clf_settings = f"{cvss_col},{fold},{granularity},{scope},{feature},{model_name}," \
	# 						   f"{'-'.join(parameters)}"
	# 			print(clf_settings)
	# 			multiclass = False if len(np.unique(y_train)) else True
	# 			# Get and evaluate the classifier
	# 			clf = get_classifier(model_name, multiclass, len(np.unique(y_train)), *parameters)
	# 			evaluate(clf, x_train, y_train, x_val, y_val, x_test, y_test,
	# 					 clf_settings, result_file, write=True)


	print(granularity, scope, cvss_col, feature)









	train_df = df[df["partition"]=="train"]
	val_df = df[df["partition"]=="valid"]
	test_df = df[df["partition"]=="test"]

	x_train = train_df[data_col].values
	x_val = val_df[data_col].values
	x_test = test_df[data_col].values

	y_train = train_df[cvss_col].values
	print(type(y_train))
	print(y_train)
	y_train = y_train.astype("int")
		
	y_val = val_df[cvss_col].values
	y_val = y_val.astype("int")
	y_test = test_df[cvss_col].values
	y_test = y_test.astype("int")
	import numpy as np
	print("============================")

	print(np.unique(y_train))
	print("============================")


	x_train, x_val, x_test = generate_features(x_train, x_val, x_test, feature)

	for model_name in models:

		param_set = []

		if model_name == 'lr':
			param_set = list(itertools.product(*[regularization_lr]))
		elif model_name == 'svm':
			param_set = list(itertools.product(*[regularization_svm]))
		elif model_name == 'knn':
			param_set = list(itertools.product(*[neighbours, weights, norms]))
		elif model_name == 'rf' or model_name == 'xgb' or model_name == 'lgbm':
			param_set = list(itertools.product(*[estimators, leaf_nodes]))
		# Run for each parameter configuration

		for parameters in param_set:
			clf_settings = f"{cvss_col},{granularity},{scope},{feature},{model_name}," \
						   f"{'-'.join(parameters)}"
			print(clf_settings)
			multiclass = True
			num_classes = len(np.unique(y_train))
			# Get and evaluate the classifier
			clf = get_classifier(model_name, multiclass, len(np.unique(y_train)), *parameters,num_classes)
			evaluate(clf, x_train, y_train, x_val, y_val, x_test, y_test,
					 clf_settings, result_file, write=True)

	print('Execution time:', time.time() - start_time, 's.')

	avg_df = compute_average_results(result_file)
	avg_df.to_csv(result_folder/ f'avg_results_{granularity}_{scope}_{cvss_col}_{feature}.csv', index=False)
	best_results = find_best_results(avg_df)
	print(best_results.values)
	best_results.to_csv(result_folder /f'best_results_{granularity}_{scope}_{cvss_col}_{feature}.csv', index=False)

	print('Done execution!!!')


# ---------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
print("----------------------------")
with open('evaluate.csv', newline='') as csvfile:
	# 创建CSV读取器对象
	reader = csv.reader(csvfile, delimiter=',')

	# 逐行读取CSV内容
	for row in reader:
		# 解包CSV行数据到变量 a, b, c, d
		a, b, c, d = row

		# 打印读取到的参数（类似于echo命令）
		print(a, b, c, d)

		# 调用主函数，传递读取到的参数
		main(a, b, c, d)
# main("method", "whole", "AV", "subwords")
