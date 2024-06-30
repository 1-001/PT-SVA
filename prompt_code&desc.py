import os
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, \
    matthews_corrcoef
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 32
num_class = 4
max_seq_l =512
lr = 5e-5
num_epochs = 5000
use_cuda = True
model_name = "codet5"
pretrainedmodel_path = "E:\models\codet5-base"  # the path of the pre-trained model
early_stop_threshold =10


from openprompt.data_utils import InputExample

classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]

def read_prompt_examples(filename):
    """Read examples from filename."""
    examples = []
    if 'train' in filename:
        data = pd.read_excel(filename).astype(str)  # .sample(frac=1)
    else:
        data = pd.read_excel(filename).astype(str)
    # data2['code'] = data2['lang'] + ':' + data2['code']
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    severity = data['severity'].tolist()
    for idx in range(len(data)):
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=int(severity[idx]),
            )
        )


    return examples


# load plm
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)
# construct hard template
from openprompt.prompts import ManualTemplate,MixedTemplate,SoftTemplate

#mix
template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability description:" {"placeholder":"text_b"} {"soft":"Classify the severity:"} {"mask"}'

mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text,model=plm)

# DataLoader
from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\\train.xlsx"), template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    batch_size=batch_size, shuffle=True,
                                    teacher_forcing=False, predict_eos_token=False, truncate_method="head",
                                    decoder_max_length=3)
validation_dataloader = PromptDataLoader(dataset=read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\\valid.xlsx"), template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         batch_size=batch_size, shuffle=True,
                                         teacher_forcing=False, predict_eos_token=False, truncate_method="head",
                                         decoder_max_length=3)
test_dataloader = PromptDataLoader(dataset=read_prompt_examples(r"C:\Users\Admin\Desktop\data_c++\\test.xlsx"), template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                   batch_size=batch_size, shuffle=False,
                                   teacher_forcing=False, predict_eos_token=False, truncate_method="head",
                                   decoder_max_length=3)

# define the verbalizer
from openprompt.prompts import ManualVerbalizer

myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={
                                    "LOW": ["low","slight"],
                                    "MEDIUM": ["medium","moderate"],
                                    "HIGH": ["high","severe"],
                                    "CRITICAL": ["critical","significant"]
                                })
from openprompt import PromptForClassification

prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
]
# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)  # learning rate for model parameters
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-5)  # learning rate for prompt parameters

num_training_steps = num_epochs * len(train_dataloader)
scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0,
                                             num_training_steps=num_training_steps)  # set warmup steps
scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0,
                                             num_training_steps=num_training_steps)  # set warmup steps

from tqdm.auto import tqdm


def test(prompt_model, test_dataloader,name):
    num_test_steps = len(test_dataloader)
    progress_bar = tqdm(range(num_test_steps))
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']

            progress_bar.update(1)
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = accuracy_score(alllabels, allpreds)
        # precision = precision_score(alllabels, allpreds)
        # recall = recall_score(alllabels, allpreds)

        precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
        precisionma, recallma, f1ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
        mcc=matthews_corrcoef(alllabels, allpreds)
        # precision, recall, f1, _ = precision_recall_fscore_support(alllabels, allpreds, average=None)
        with open(os.path.join('./results',  "{}.pred.csv".format(name)), 'w',
                  encoding='utf-8') as f, \
                open(os.path.join('./results', "{}.gold.csv".format(name)), 'w',
                     encoding='utf-8') as f1:

            for ref, gold in zip(allpreds, alllabels):
                f.write(str(ref) + '\n')
                f1.write(str(gold) + '\n')

        print("acc: {}   precisionma :{}  recallma:{} recallwei{} weighted-f1: {}  macro-f1: {} mcc:{}".format(acc, precisionma, recallma, recallwei, f1wei,
                                                                                         f1ma,mcc))
    return acc, precisionma, recallma, f1wei, f1ma


output_dir = "result log"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

progress_bar = tqdm(range(num_training_steps))
bestmetric = 0
bestepoch = 0
early_stop_count = 0
for epoch in range(num_epochs):
    # train
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):

        if use_cuda:
            inputs = inputs.cuda()


        logits = prompt_model(inputs)

        labels = inputs['tgt_text'].cuda()



        loss = loss_func(logits, labels)
        try:
            loss.backward()
        except:
            print(loss)
            exit()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        scheduler1.step()
        optimizer2.step()
        optimizer2.zero_grad()
        scheduler2.step()
        progress_bar.update(1)
    print("\nEpoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
    this_epoch_best = False

    # validate
    print('\n\nepoch{}------------validate------------'.format(epoch))
    acc, precision, recall, f1wei, f1mi = test(prompt_model, validation_dataloader,name="dev")
    if f1mi > bestmetric:
        bestmetric = f1mi
        bestepoch = epoch
        this_best_epoch=True
        torch.save(prompt_model.state_dict(), f"{output_dir}/best.ckpt")
    # if this_epoch_best:
    #     early_stop_count=0
    else:
        early_stop_count += 1
        if early_stop_count == early_stop_threshold:
            print("early stopping!!!")
            break
    # test
print('\n\nepoch{}------------test------------'.format(epoch))
prompt_model.load_state_dict(torch.load(os.path.join("E:\wjy\SVA\SVA\model_code\\result log", "best.ckpt"), map_location=torch.device('cuda:0')))
acc, precisionma, recallma, f1wei, f1ma = test(prompt_model, test_dataloader,name="test")