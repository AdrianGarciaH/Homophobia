from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import numpy as np
import torch


class dataset(Dataset):
    
  def __init__(self, all_data, tokenizer, labels_to_ids, max_len, for_training):
        self.len = len(all_data)
        self.data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.for_training = for_training


  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        #tweet_id = self.data[index][0]
        sentence = self.data[index][0]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        #item['tweet_id'] = tweet_id
        item['orig_sentence'] = sentence

        # step 4: if it is for training, get input labels as well
        if self.for_training:
            input_label = self.data[index][1]
            labels = self.labels_to_ids[input_label]
            item['labels'] = torch.as_tensor(labels)

        return item

  def __len__(self):
        return self.len



def initialize_data(tokenizer, initialization_input, input_data, labels_to_ids, shuffle = True):
    max_len, batch_size = initialization_input
    data_split = dataset(input_data, tokenizer, labels_to_ids, max_len, True)


    params = {'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': 4
                }

    loader = DataLoader(data_split, **params)
    

    return loader

def initialize_test(tokenizer, initialization_input, input_data, labels_to_ids, shuffle = True):
    max_len, batch_size = initialization_input
    data_split = dataset(input_data, tokenizer, labels_to_ids, max_len, False)


    params = {'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': 4
                }

    loader = DataLoader(data_split, **params)

    return loader




if __name__ == '__main__':
  pass