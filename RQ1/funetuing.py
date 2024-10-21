# coding=utf-8
from __future__ import absolute_import
import os
import torch
import random
import pandas as pd
import logging
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import RobertaTokenizer, T5EncoderModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename="ft.log",
                    filemode='w',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):

    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):

    desc_prefix = "<desc>"
    data = pd.read_excel(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    severity = data['severity'].tolist()
    examples = []
    for idx in range(len(data)):
        examples.append(
            Example(
                idx=idx,
                source=' '.join(code[idx].split(' ')[:384]) + desc_prefix + ' '.join(desc[idx].split(' ')[:64]),
                target=int(severity[idx]),  
            )
        )
    return examples

class InputFeatures(object):


    def __init__(self, example_id, source_ids, target_id, source_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_id = target_id  
        self.source_mask = source_mask

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
     
        source_tokens = tokenizer(example.source, max_length=args.max_source_length, padding='max_length', truncation=True)
        source_ids = source_tokens['input_ids']
        source_mask = source_tokens['attention_mask']

        target_id = example.target  
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            logger.info("source_tokens: {}".format(source_tokens))
            logger.info("source_ids: {}".format(source_ids))
            logger.info("source_mask: {}".format(source_mask))
            logger.info("target_id: {}".format(target_id))

        features.append(
            InputFeatures(
                example_id=example.idx,
                source_ids=source_ids,
                target_id=target_id,  
                source_mask=source_mask
            )
        )
    return features

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CodeT5Classifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(CodeT5Classifier, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.encoder.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        cls_output = last_hidden_state[:, 0, :]  
        logits = self.classifier(cls_output)
        return logits

def evaluate(model, dataloader, device):

    model.eval()
    predictions, true_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        source_ids, source_mask, target_ids = [b.to(device) for b in batch]

        with torch.no_grad():
            logits = model(input_ids=source_ids, attention_mask=source_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(target_ids.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    mcc = matthews_corrcoef(true_labels, predictions)

    return accuracy, precision, recall, f1, mcc, predictions, true_labels

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default='codet5', type=str, required=False)
    parser.add_argument("--model_name_or_path", default=r'E:\models\codet5-base', type=str, required=False)  
    parser.add_argument("--output_dir", default='./output_dir_codet5_100', type=str, required=False)
    parser.add_argument("--train_filename", default=r"C:\Users\Admin\Desktop\data\\train.xlsx", type=str)
    parser.add_argument("--dev_filename", default=r"C:\Users\Admin\Desktop\data\\valid.xlsx", type=str)
    parser.add_argument("--test_filename", default=r"C:\Users\Admin\Desktop\data\\test.xlsx", type=str)

    parser.add_argument("--max_source_length", default=512, type=int)
    parser.add_argument("--do_train", default=True, action='store_true')
    parser.add_argument("--do_eval", default=True, action='store_true')
    parser.add_argument("--do_test", default=True, action='store_true')
    parser.add_argument("--train_batch_size", default=16, type=int) 
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)  
    parser.add_argument("--num_train_epochs", default=500, type=int)  
    parser.add_argument("--early_stopping_patience", default=10, type=int)  
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()


    set_seed(args.seed)


    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = CodeT5Classifier(args.model_name_or_path, num_labels=4) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    if args.do_train:
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')

        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_id for f in train_features], dtype=torch.long)  

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

      
        best_f1 = 0
        early_stop_counter = 0

        model.train()
        for epoch in range(args.num_train_epochs):
            total_loss = 0
            model.train() 

            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
                source_ids, source_mask, target_ids = [b.to(device) for b in batch]

                optimizer.zero_grad()

                logits = model(input_ids=source_ids, attention_mask=source_mask)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, target_ids)

                loss.backward()
                total_loss += loss.item()

                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} finished with average loss: {avg_loss}")

            if args.do_eval:
                dev_examples = read_examples(args.dev_filename)
                dev_features = convert_examples_to_features(dev_examples, tokenizer, args, stage='dev')

                all_source_ids = torch.tensor([f.source_ids for f in dev_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in dev_features], dtype=torch.long)
                all_target_ids = torch.tensor([f.target_id for f in dev_features], dtype=torch.long)

                dev_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids)
                dev_sampler = SequentialSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

                accuracy, precision, recall, f1, mcc, predictions, true_labels = evaluate(model, dev_dataloader, device)

            
                pd.DataFrame({"Predictions": predictions, "True Labels": true_labels}).to_csv(os.path.join(args.output_dir, "dev_predictions.csv"), index=False)

                logger.info(f"Validation Accuracy: {accuracy}")
                logger.info(f"Validation Precision: {precision}, Recall: {recall}, F1: {f1}, MCC: {mcc}")
                print(f"Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, MCC: {mcc}")

     
                if f1 > best_f1:
                    best_f1 = f1
                    early_stop_counter = 0
           
                    output_model_file = os.path.join(args.output_dir, "best_model.bin")
                    torch.save(model.state_dict(), output_model_file)  
                    tokenizer.save_pretrained(args.output_dir)
                else:
                    early_stop_counter += 1
                    logger.info(f"No improvement in F1. Early stopping counter: {early_stop_counter}")

                if early_stop_counter >= args.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break


    if args.do_test:
        
        model = CodeT5Classifier(args.model_name_or_path, num_labels=4)
        output_model_file = os.path.join(args.output_dir, "best_model.bin")
        model.load_state_dict(torch.load(output_model_file, map_location=device))  
        model.to(device)

        test_examples = read_examples(args.test_filename)
        test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')

        all_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        accuracy, precision, recall, f1, mcc, predictions, true_labels = evaluate(model, test_dataloader, device)

        
        pd.DataFrame({"Predictions": predictions, "True Labels": true_labels}).to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

        logger.info(f"Test Accuracy: {accuracy}")
        logger.info(f"Test Precision: {precision}, Recall: {recall}, F1: {f1}, MCC: {mcc}")
        print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, MCC: {mcc}")

if __name__ == "__main__":
    main()
