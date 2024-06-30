import pandas as pd
import time
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent)))
import gcn
import numpy as np
from transformers import  AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader

def filter_code(vuln_code):
	code_lines = []

	for code_line in vuln_code:
		if '//' in code_line:
			code_line = code_line[:code_line.find('//')]
		elif '/*' in code_line and '*/' in code_line:
			start_comment_index = code_line.find('/*')
			end_comment_index = code_line.find('*/')

			code_line = code_line[:start_comment_index] + code_line[end_comment_index + 2:]

		code_lines.append(code_line)

	return '\n'.join(code_lines)


def extract_clean_code(row, output='code'):
    # Output options: code, context and all (code + context)
	code = np.asarray(row['code'].splitlines())
	res = ""
	if output == 'code':
		vul_lines = np.asarray([int(line) for line in row["vul_line"].split()])-1
		res = code[vul_lines]
	elif output == 'context':
		vul_lines = np.asarray([int(line) for line in row["vul_line"].split()])-1
		method_lines = np.asarray(list(range(len(code))))
		method_lines = method_lines.tolist()
		vul_lines = vul_lines.tolist()
		code_lines = np.asarray(
			list(set(method_lines)  - set(vul_lines)))
		if len(code_lines)!=0:
			res = code[code_lines]
		else:
			res = ""
	elif output == 'all':
		res = code
	return filter_code(res)

def extract_method_vuln_code(row):
	code = np.asarray(row['code'].splitlines())

	vul_lines = np.asarray([int(line) for line in row["vul_line"].split()])-1

	vuln_code = code[vul_lines]

	return filter_code(vuln_code)

def extract_context_scope(row, scope_size=5):
	#return context_code_lines(The index starts at 1)
	start_line = 0
	end_line = len(row['code'].splitlines())-1

	context_lines = []
	vul_lines = np.asarray([int(line) for line in row["vul_line"].split()])
	context_lines = vul_lines.tolist()


	for line in vul_lines:
		start_scope = line - scope_size

		if start_scope < start_line:
			start_scope = start_line

		end_scope = line + scope_size

		if end_scope > end_line:
			end_scope = end_line

		context_lines.extend([line_index for line_index in range(start_scope, end_scope + 1)])

	return sorted(list(set(context_lines)))

def extract_surrounding_context_code(row):
	
	code = np.asarray(row['code'].splitlines())
	# print(row['surrounding_context'])
	# print(set(row['surrounding_context']))
	vuln_lines = np.asarray(list(set(row['surrounding_context'])))-1
	if len(vuln_lines) == 0:
		return ''

	vuln_code = code[vuln_lines]

	return filter_code(vuln_code)

def extract_surrounding_context_code_wo_vuln(row, granularity):
	
	code = np.asarray(row['code'].splitlines())
	vul_lines = np.asarray([int(line) for line in row["vul_line"].split()])
	vul_lines = np.asarray(list(
		set(row['surrounding_context']) - set(vul_lines) ))-1

	if len(vul_lines) == 0:
		return ''

	vul_lines = code[vul_lines]

	return filter_code(vul_lines)
	
def extract_random_context(row):

	code = np.asarray(row['code'].splitlines())
	vul_lines = np.asarray([int(line) for line in row["vul_line"].split()])-1

	method_lines = np.asarray(list(range(len(code)))) + 1
	method_lines = method_lines.tolist()
	code_lines = np.asarray(
		list(set(method_lines)   - set(vul_lines))) - 1

	vuln_lines_len = len(vul_lines)
	if vuln_lines_len > len(code_lines):
		vuln_lines_len = len(code_lines)

	code_lines = np.random.RandomState(vuln_lines_len).choice(code_lines, vuln_lines_len, replace=False)

	if len(code_lines) == 0:
		return ''

	code = code[code_lines]

	return filter_code(code)
def extract_codebert_features(text, model, tokenizer, batch_size=64):
	start_time = time.time()
	dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(dev)
	text = text.tolist()
	max_length = 512
	tokens_ids = tokenizer(text, max_length=max_length, padding=True, truncation=True, add_special_tokens=True)

	# print(tokens_ids)

	attention_masks = tokens_ids['attention_mask']
	tokens_ids = tokens_ids['input_ids']

	batch_size = 32

	# wrap tensors
	train_data = TensorDataset(torch.tensor(tokens_ids), torch.tensor(attention_masks),
							   torch.tensor([1] * len(tokens_ids)))

	# dataLoader for train set
	train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)

	# for tokens in tokens_ids:
	# 	print(tokenizer.decode(tokens))

	features = []

	for step, batch in enumerate(train_dataloader):

		# print('Step:', step + 1)

		sent_ids, masks, labels = batch
		sent_ids = sent_ids.to(dev)
		masks = masks.to(dev)

		if step == 0:
			features = model(sent_ids, attention_mask=masks)[0][:, 0, :].squeeze().detach().cpu().numpy().tolist()
		else:
			features.extend(model(sent_ids, attention_mask=masks)[0][:, 0, :].squeeze().detach().cpu().numpy().tolist())

	# print(len(features), len(features[0]))

	print('Execution time:', time.time() - start_time, 's.')

	return features

filename = "C:/Users/Admin/Desktop/FVA-master/Data/function_level_Le_Data/data.xlsx"
df_method = pd.read_excel(filename)
df_method = df_method.rename(columns={"delete_lines":"vul_line"})

# n_folds = 10

# method_map = create_fold(df_method, 'id', folds=n_folds)

# method_map.to_csv(Data_folder / 'method_map.csv', index=False)



# cvss_cols = ['AV', 'AC', 'PR', 'UI', 'S', 'C','I','A', 'cvss3_severity']
cvss_cols = ['cvss3_severity']

cvss_cols += ["partition"]
print('Loaded data')
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
print(len(df_method))
Data_folder = gcn.get_dir(gcn.data_dir() / "function_level_Le_Data" ) 
#############################################################################################

# Whole method
# selected_cols = ['method_change_id', 'code', 'noisy_lines', 'start_line']
selected_cols = ['id', 'func_before']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['filtered_code'] = df_tmp[['code']].apply(
	lambda r: extract_clean_code(r, 'all'), axis=1)

df_tmp = df_tmp.drop(columns=["code"])
df_tmp = df_tmp.rename(columns={'id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
print(df_tmp)

df_tmp.to_csv(Data_folder / 'method_whole_codebert.csv', index=False)

# Vuln lines without context in methods
selected_cols = ['id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['vuln_code'] = df_tmp[['code', 'vul_line']].apply(lambda r: extract_method_vuln_code(r), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'vul_line'])
df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'id': 'key'}).reset_index(drop=True)

codebert_features_code = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features_code

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_lines_without_context_codebert.csv', index=False)


# program slice
# Vuln lines with context in methods
# selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'noisy_lines']
# selected_cols.extend(cvss_cols)
# df_tmp = df_method[selected_cols].copy()
# df_tmp['vuln_code'] = df_tmp[['code', 'context_lines', 'start_line', 'noisy_lines']].apply(lambda r: extract_context_code_method(r), axis=1)
# df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'noisy_lines'])
# df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)

# codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
# df_tmp['codebert'] = codebert_features

# df_tmp = df_tmp.drop(columns=['code'])

# print(len(df_tmp), df_tmp.columns)
# df_tmp.to_parquet('Data/method_lines_with_all_context_codebert.parquet', index=False)

#
# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods
scope_size = 6

selected_cols = ['id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['surrounding_context'] = df_tmp[["code",'vul_line']].apply(
	lambda r: extract_context_scope(r, scope_size=scope_size), axis=1)

df_tmp['context_code'] = df_tmp[['code', 'surrounding_context']].apply(
	lambda r: extract_surrounding_context_code(r), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line',  'surrounding_context'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'id': 'key'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_lines_with_surrounding_context_codebert.csv', index=False)

# Context only in methods
selected_cols = ['id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['filtered_code'] = df_tmp[['code', 'vul_line']].apply(
	lambda r: extract_clean_code(r,  'context'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line'])
df_tmp = df_tmp.rename(columns={'id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder /'method_context_only_codebert.csv', index=False)

# program slicing
# Non vuln lines in methods (all scope - program slicing scope)
# selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'noisy_lines']
# selected_cols.extend(cvss_cols)
# df_tmp = df_method[selected_cols].copy()
# df_tmp['vuln_code'] = df_tmp[['code', 'context_lines', 'start_line', 'noisy_lines']].apply(
# 	lambda r: extract_non_vuln_code_method(r), axis=1)
# df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'noisy_lines'])
# df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)

# codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
# df_tmp['codebert'] = codebert_features

# df_tmp = df_tmp.drop(columns=['code'])

# print(len(df_tmp), df_tmp.columns)
# df_tmp.to_parquet('Data/method_non_vuln_codebert.parquet', index=False)

###########################

# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods
scope_size = 6

selected_cols = ['id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['surrounding_context'] = df_tmp[["code",'vul_line']].apply(
	lambda r: extract_context_scope(r, scope_size=scope_size), axis=1)

df_tmp['context_code'] = df_tmp[['code', 'surrounding_context', 'vul_line']].apply(
	lambda r: extract_surrounding_context_code_wo_vuln(r, granularity='method'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line', 'surrounding_context'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'id': 'key'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_surrounding_only_codebert.csv', index=False)

###########################
#program slice
# Vuln lines with context in methods
# selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines']
# selected_cols.extend(cvss_cols)
# df_tmp = df_method[selected_cols].copy()

# df_tmp['context_code'] = df_tmp[['code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines']].apply(
# 	lambda r: extract_context_code_method_wo_vuln(r), axis=1)

# df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines'])
# df_tmp = df_tmp.rename(columns={'context_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)

# codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
# df_tmp['codebert'] = codebert_features

# df_tmp = df_tmp.drop(columns=['code'])

# print(len(df_tmp), df_tmp.columns)
# df_tmp.to_parquet('Data/method_slicing_only_codebert.parquet', index=False)

# Context only with the same size as the vulnerable statements in methods
selected_cols = ['id', 'func_before',  'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['filtered_code'] = df_tmp[['code',  'vul_line']].apply(
	lambda r: extract_random_context(r), axis=1)

df_tmp = df_tmp.drop(columns=['code' , 'vul_line'])
df_tmp = df_tmp.rename(columns={'method_change_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_random_context_codebert.csv', index=False)