import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer,MixedTemplate
from openprompt import PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 32
num_class = 4
max_seq_l = 512
lr = 5e-5
num_epochs = 5000
use_cuda = True
model_name = "roberta"
pretrainedmodel_path = "E:\\models\\CodeBERT"
early_stop_threshold = 10

classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def read_data_to_dataframe(filename):
    data = pd.read_excel(filename).astype(str)
    return data[['abstract_func_before', 'description', 'severity']]

def convert_dataframe_to_dataset(data):
    examples = {
        'text_a': [],
        'text_b': [],
        'label': []
    }
    for idx, row in data.iterrows():
        examples['text_a'].append(' '.join(row['abstract_func_before'].split(' ')[:384]))
        examples['text_b'].append(' '.join(row['description'].split(' ')[:64]))
        examples['label'].append(int(row['severity']))
    return Dataset.from_dict(examples)

# Read and convert data
train_data = read_data_to_dataframe(r"C:\Users\Admin\Desktop\data_c++\train.xlsx")
valid_data = read_data_to_dataframe(r"C:\Users\Admin\Desktop\data_c++\\valid.xlsx")
test_data = read_data_to_dataframe(r"C:\Users\Admin\Desktop\data_c++\\test.xlsx")

train_dataset = convert_dataframe_to_dataset(train_data)
valid_dataset = convert_dataframe_to_dataset(valid_data)
test_dataset = convert_dataframe_to_dataset(test_data)

# Create the splits dictionary
train_val_test = {
    'train': train_dataset,
    'validation': valid_dataset,
    'test': test_dataset
}

# Convert to InputExample format
dataset = {}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in train_val_test[split]:
        input_example = InputExample(text_a=data['text_a'], text_b=data['text_b'], label=data['label'])
        dataset[split].append(input_example)

# Load PLM
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", pretrainedmodel_path)

# Construct template
template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability description: {"placeholder":"text_b"} {"soft":"Classify the severity:"} {"mask"}'
mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text,model=plm)

# DataLoaders
train_dataloader = PromptDataLoader(dataset=dataset['train'], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                    predict_eos_token=False, truncate_method="head", decoder_max_length=3)
validation_dataloader = PromptDataLoader(dataset=dataset['validation'], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                         predict_eos_token=False, truncate_method="head", decoder_max_length=3)
test_dataloader = PromptDataLoader(dataset=dataset['test'], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                   batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                   predict_eos_token=False, truncate_method="head", decoder_max_length=3)

# Verbalizer
myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={"LOW": ["low", "slight"],
                                             "MEDIUM": ["medium", "moderate"],
                                             "HIGH": ["high", "severe"],
                                             "CRITICAL": ["critical", "significant"]})

# Prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

# Optimizers and scheduler
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
]
optimizer_grouped_parameters2 = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)

# Test function
def test(prompt_model, test_dataloader, name):
    num_test_steps = len(test_dataloader)
    progress_bar = tqdm(range(num_test_steps))
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']

            progress_bar.update(1)
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = accuracy_score(alllabels, allpreds)
        precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
        precisionma, recallma, f1ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
        mcc = matthews_corrcoef(alllabels, allpreds)
        with open(os.path.join('./results',  "{}.pred.csv".format(name)), 'w', encoding='utf-8') as f, \
                open(os.path.join('./results', "{}.gold.csv".format(name)), 'w', encoding='utf-8') as f1:
            for ref, gold in zip(allpreds, alllabels):
                f.write(str(ref) + '\n')
                f1.write(str(gold) + '\n')

        print("acc: {}   precisionma :{}  recallma:{} recallwei{} weighted-f1: {}  macro-f1: {} mcc:{}".format(acc, precisionma, recallma, recallwei, f1wei, f1ma, mcc))
    return acc, precisionma, recallma, f1wei, f1ma

# Training and evaluation
output_dir = "vultypeprompt3_log"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

progress_bar = tqdm(range(num_training_steps))
bestmetric = 0
bestepoch = 0
early_stop_count = 0
for epoch in range(num_epochs):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label'].cuda()

        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        scheduler1.step()
        optimizer2.step()
        optimizer2.zero_grad()
        scheduler2.step()
        progress_bar.update(1)
    print("\nEpoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

    print('\n\nepoch{}------------validate------------'.format(epoch))
    acc, precision, recall, f1wei, f1ma = test(prompt_model, validation_dataloader, name="dev")
    if f1ma > bestmetric:
        bestmetric = f1ma
        bestepoch = epoch
        torch.save(prompt_model.state_dict(), f"{output_dir}/best.ckpt")
    else:
        early_stop_count += 1
        if early_stop_count == early_stop_threshold:
            print("early stopping!!!")
            break

print('\n\nepoch{}------------test------------'.format(epoch))
prompt_model.load_state_dict(torch.load(os.path.join(output_dir, "best.ckpt"), map_location=torch.device('cuda:0')))
acc, precisionma, recallma, f1wei, f1ma = test(prompt_model, test_dataloader, name="test")
