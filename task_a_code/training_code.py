#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from load_data import initialize_data
from reading_datasets import read_task5
from labels_to_ids import labels_to_ids
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train(epoch, train_loader, model, optimizer, device, grad_step = 1, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    tr_precision, tr_recall = 0, 0
    tr_f1score = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    optimizer.zero_grad()
    
    for idx, batch in enumerate(train_loader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        if (idx + 1) % 20 == 0:
            print('FINSIHED BATCH:', idx, 'of', len(train_loader))

        #loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += output[0]

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        
        # Compute Precision
        tmp_tr_precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0,1,2], average = None, zero_division=0 )[2]
        tr_precision += tmp_tr_precision
        
        # Compute Recall
        tmp_tr_recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0,1,2], average = None, zero_division=0 )[2]
        tr_recall += tmp_tr_recall
        
        # Compute f1score
        tmp_tr_f1score = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(),labels=[0,1,2], average=None, zero_division=0)[2]
        tr_f1score += tmp_tr_f1score
    

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )
        
        # backward pass
        output['loss'].backward()
        if (idx + 1) % grad_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    tr_precision = tr_precision / nb_tr_steps
    tr_recall = tr_recall / nb_tr_steps
    tr_f1score= tr_f1score / nb_tr_steps
    #print(f"Training loss epoch: {epoch_loss}")
    #print(f"Training accuracy epoch: {tr_accuracy}")

    return model


# In[3]:

'''
def testing(model, testing_loader, labels_to_ids, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    eval_precision, eval_recall = 0, 0
    eval_f1score = 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
     
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            eval_loss += output['loss'].item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy
            
            # Compute Precision
            tmp_eval_precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0,1,2], average = None, zero_division=0)[2]
            eval_precision += tmp_eval_precision
            
            # Compute Recall
            tmp_eval_recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0,1,2], average = None, zero_division=0)[2]
            eval_recall += tmp_eval_recall
            
            # Compute f1score
            tmp_eval_f1score = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(),labels=[0,1,2], average=None, zero_division=0)[2]
            eval_f1score += tmp_eval_f1score

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    eval_precision = eval_precision / nb_eval_steps
    eval_recall = eval_recall / nb_eval_steps
    eval_f1score = eval_f1score / nb_eval_steps
    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions, eval_accuracy, eval_precision, eval_recall, eval_f1score
'''

def testing(model, testing_loader, labels_to_ids, device):
    print("TESTING DATA")
    # put model in evaluation mode
    torch.no_grad()
    
    nb_eval_steps = 0
    eval_preds = []

    eval_tweet_ids, eval_orig_sentences = [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            
            # to attach back to prediction data later 
            tweet_ids = batch['tweet_id']
            orig_sentences = batch['orig_sentence']

            output = model(ids, attention_mask=mask)
            
            # eval_loss += output['loss'].item()

            nb_eval_steps += 1
            # nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                print(f"Went through 100 steps")

            predictions = torch.argmax(output.logits, axis = 1)

            eval_preds.extend(predictions)

            eval_tweet_ids.extend(tweet_ids)
            eval_orig_sentences.extend(orig_sentences)

    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    overall_prediction_data = pd.DataFrame(zip(eval_tweet_ids, eval_orig_sentences, predictions), columns=['tweet_id', 'text', 'label'])

    return overall_prediction_data

# In[4]:


def main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location):
    #Initialization training parameters
    max_len = 256
    batch_size = 32
    grad_step = 1
    learning_rate = 1e-05
    initialization_input = (max_len, batch_size)

    #Reading datasets and initializing data loaders
    dataset_location = '../2022.07.07_task5/'

    train_data = read_task5(dataset_location , split = 'train')
    dev_data = read_task5(dataset_location , split = 'dev')
    #test_data = read_task5(dataset_location , split = 'dev')#load test set
    labels_to_ids = task5_labels_to_ids
    #input_data = (train_data, dev_data, labels_to_ids)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    if model_load_flag:
        tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        model = AutoModelForSequenceClassification.from_pretrained(model_load_location)
    else: 
        tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader = initialize_data(tokenizer, initialization_input, train_data, labels_to_ids, shuffle = True)
    dev_loader = initialize_data(tokenizer, initialization_input, dev_data, labels_to_ids, shuffle = True)
    #test_loader = initialize_data(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = True)#create test loader

    best_dev_acc = 0
    best_test_acc = 0
    best_dev_precision = 0
    best_test_precision = 0
    best_dev_recall = 0
    best_test_recall = 0
    best_dev_f1score = 0
    best_test_f1score = 0
    best_epoch = -1
    
    list_dev_acc = [] 
    list_test_acc = []  
    list_dev_precision = []  
    list_test_precision  = []  
    list_dev_recall = []  
    list_test_recall = []  
    list_dev_f1score = []  
    list_test_f1score = []
    
    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train model
        if not model_load_flag:
            model = train(epoch, train_loader, model, optimizer, device, grad_step)
        
        #testing and logging
        labels_dev, predictions_dev, dev_accuracy, dev_precision, dev_recall, dev_f1score = testing(model, dev_loader, labels_to_ids, device)
        print('DEV ACC:', dev_accuracy)
        print('DEV Precision:' , dev_precision)
        print('DEV Recall:' , dev_recall)
        print('DEV F1Score:' , dev_f1score)
        
        list_dev_acc.append(dev_accuracy)     
        list_dev_precision.append(dev_precision)   
        list_dev_recall.append(dev_recall)  
        list_dev_f1score.append(dev_f1score)  
        
        
        #labels_test, predictions_test, test_accuracy, test_precision, test_recall, test_f1score = testing(model, test_loader, labels_to_ids, device)
        #print('TEST ACC:', test_accuracy)
        #print('TEST Precision:' , test_precision)
        #print('TEST Recall:' , test_recall)
        #print('TEST F1Score:' , test_f1score)
        
        #list_test_acc.append(test_accuracy) 
        #list_test_precision.append(test_precision)  
        #list_test_recall.append(test_recall)
        #list_test_f1score.append(test_f1score) 

        #saving model
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            #best_test_acc = test_accuracy
        if dev_precision > best_dev_precision:
            best_dev_precision = dev_precision
            #best_test_precision = test_precision
        if dev_recall > best_dev_recall:
            best_dev_recall = dev_recall
            #best_test_recall = test_recall
        if dev_f1score > best_dev_f1score:
            best_dev_f1score = dev_f1score
            #best_test_f1score = test_f1score
            best_epoch = epoch
            
            if model_save_flag:
                os.makedirs(model_save_location, exist_ok=True)
                tokenizer.save_pretrained(model_save_location)
                model.save_pretrained(model_save_location)

        now = time.time()
        print('BEST ACCURACY --> ', 'DEV:', round(best_dev_acc, 5))
        print('BEST PRECISION --> ', 'DEV:', round(best_dev_precision, 5))
        print('BEST RECALL --> ', 'DEV:', round(best_dev_recall, 5))
        print('BEST F1SCORE --> ', 'DEV:', round(best_dev_f1score, 5))
        print('TIME PER EPOCH:', (now-start)/60 )
        print()

    return best_dev_acc, best_test_acc, best_epoch, best_dev_precision, best_test_precision, best_dev_recall, best_test_recall, best_dev_f1score, best_test_f1score, list_dev_acc, list_test_acc, list_dev_precision, list_test_precision, list_dev_recall, list_test_recall, list_dev_f1score, list_test_f1score


# In[5]:


if __name__ == '__main__':
    n_epochs = 10
    models = ['bert-base-multilingual-uncased']
    
    #model saving parameters
    model_save_flag = False
    model_load_flag = False
    
    overall_list_dev_acc = [] 
    overall_list_test_acc = []    
    overall_list_dev_precision = []  
    overall_list_test_precision  = []  
    overall_list_dev_recall = []  
    overall_list_test_recall = []  
    overall_list_dev_f1score = []  
    overall_list_test_f1score = [] 
    
    for i in range(5):
        
        for model_name in models:

            model_save_location = 'saved_models/' + model_name + str(i)
            model_load_location = None

            best_dev_acc, best_test_acc, best_epoch, best_dev_precision, best_test_precision, best_dev_recall, best_test_recall, best_dev_f1score, best_test_f1score, list_dev_acc, list_test_acc, list_dev_precision, list_test_precision, list_dev_recall, list_test_recall, list_dev_f1score, list_test_f1score = main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location)
            
            overall_list_dev_acc.append(list_dev_acc) 
            overall_list_test_acc.append(list_test_acc) 
            overall_list_dev_precision.append(list_dev_precision)  
            overall_list_test_precision.append(list_test_precision) 
            overall_list_dev_recall.append(list_dev_recall)  
            overall_list_test_recall.append(list_test_recall)  
            overall_list_dev_f1score.append(list_dev_f1score)  
            overall_list_test_f1score.append(list_test_f1score) 





