import os, sys, random, pickle, argparse, yaml, jsonlines
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from transformers import T5Tokenizer

import kss

torch.set_printoptions(threshold = float('inf'))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class Text2MusicDataset(Dataset):
    def __init__(self, configs, captions, remi_tokenizer, dataset_path, mode='train', shuffle = False):
        self.mode = mode
        self.captions = captions

        if shuffle:
            random.shuffle(self.captions)

        self.dataset_path = dataset_path
        # configs['raw_data']['raw_data_folders']['commu']['folder_path']

        self.remi_tokenizer = remi_tokenizer

        self.t5_tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-base-ko')

        self.decoder_max_sequence_length = configs['model']['decoder_max_sequence_length']

        print('Length of dataset: ', len(self.captions))

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        midi_filepath = os.path.join(self.dataset_path, self.captions[idx]['location'])
        # rint(f"[INFO] MIDI file loaded: {midi_filepath}")

        tokens = self.remi_tokenizer(midi_filepath)

        if len(tokens.ids) == 0:
            tokenized_midi = [self.remi_tokenizer['BOS_None'], self.remi_tokenizer['EOS_None']]
        else:
            tokenized_midi = [self.remi_tokenizer['BOS_None']] + tokens.ids + [self.remi_tokenizer['EOS_None']]

        # Drop a random number of sentences from the caption
        do_drop = random.random() > 0.5
        if do_drop:
            sentences = kss.split_sentences(caption)
            # print(sentences)

            sent_length = len(sentences)
            if sent_length < 4:
                how_many_to_drop = int(np.floor((20 + random.random() * 30) / 100 * sent_length))
            else:
                how_many_to_drop = int(np.ceil((20 + random.random() * 30) / 100 * sent_length))

            which_to_drop = np.random.choice(sent_length, how_many_to_drop, replace = False)
            # print(which_to_drop.tolist())

            new_sentences = ' '.join(sentences[i] for i in range(sent_length) if i not in which_to_drop.tolist())
        else:
            new_sentences = caption

        inputs = self.t5_tokenizer(new_sentences, return_tensors = 'pt', padding = True, truncation = True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if len(tokenized_midi) < self.decoder_max_sequence_length:
            labels = F.pad(torch.tensor(tokenized_midi), (0, self.decoder_max_sequence_length - len(tokenized_midi))).to(torch.int64)
        else:
            labels = torch.tensor(tokenized_midi[0:self.decoder_max_sequence_length]).to(torch.int64)

        return input_ids, attention_mask, labels