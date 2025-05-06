#!/usr/bin/env python

# 2022 Dongji Gao
# 2022 Yiwen Shao

import os
import sys
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import AsrDataset
from model import LSTM_ASR
import matplotlib.pyplot as plt
import json
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
words_path = "data/clsp.trnscr"
with open(words_path, 'r') as f:
    WORDS = sorted(set(line.strip() for i, line in enumerate(f) if i > 0))
TOKENIZED_WORDS = [torch.tensor([26] + [ord(c) - ord('a') for c in word] + [26]) for word in WORDS]
TOKENIZED_WORDS = [tokenized_word.to(device) for tokenized_word in TOKENIZED_WORDS]

def collate_fn(batch):
    """
    This function will be passed to your dataloader.
    It pads word_spelling (and features) in the same batch to have equal length.with 0.
    :param batch: batch of input samples
    :return: (recommended) padded_word_spellings, 
                           padded_features,
                           list_of_unpadded_word_spelling_length (for CTCLoss),
                           list_of_unpadded_feature_length (for CTCLoss)
    """
    # === write your code here ===

    tokenized_words, features = zip(*batch)

    word_lengths = torch.tensor([len(word) for word in tokenized_words])
    feature_lengths = torch.tensor([len(feature) for feature in features])

    tokenized_words = pad_sequence(tokenized_words, batch_first=True, padding_value=27)
    features = pad_sequence(features, batch_first=True)

    return tokenized_words, features, word_lengths, feature_lengths


def train(train_dataloader, model, ctc_loss, optimizer, device):
    # === write your code here ===
    model.train()
    loss_sum = 0
    loss_count = 0
    for batch in train_dataloader:
        tokenized_words, features, word_lengths, feature_lengths = batch
        tokenized_words = tokenized_words.to(device)
        features = features.to(device)
        optimizer.zero_grad()
        logprobs = model(features)
        logprobs = logprobs.transpose(0, 1)


        loss = ctc_loss(logprobs, tokenized_words, feature_lengths, word_lengths)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        loss_count += 1
    return loss_sum / loss_count

def get_dev_loss(dev_dataloader, model, ctc_loss, device):
    model.eval()
    loss_sum = 0
    loss_count = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            tokenized_words, features, word_lengths, feature_lengths = batch
            tokenized_words = tokenized_words.to(device)
            features = features.to(device)
            logprobs = model(features)
            logprobs = logprobs.transpose(0, 1)
            loss = ctc_loss(logprobs, tokenized_words, feature_lengths, word_lengths)
            loss_sum += loss.item()
            loss_count += 1
    return loss_sum / loss_count


def decode(batch, model, ctc_loss, device):
    # === write your code here ===
    _, features, _, feature_lengths = batch
    features = features.to(device)
    model.eval()
    with torch.no_grad():
        logprobs = model(features)
        logprobs = logprobs.transpose(0, 1)
        loss_mat = torch.zeros(len(TOKENIZED_WORDS), len(features))
        for i, tokenized_word in enumerate(TOKENIZED_WORDS):
            tokenized_word_tile = tokenized_word.tile(len(features), 1)
            word_lengths = torch.tensor([len(tokenized_word)] * len(features))
            loss = ctc_loss(logprobs, tokenized_word_tile, feature_lengths, word_lengths)
            loss_mat[i] = loss
        loss_mat_negative = -loss_mat
        loss_mat_softmax = nn.functional.softmax(loss_mat_negative, dim = 0)
        probs, decoded_word_indices = torch.max(loss_mat_softmax, dim = 0)
        decoded_words = [WORDS[i] for i in decoded_word_indices]
    return probs, decoded_words

def predict_with_probs(test_dataloader, model, ctc_loss, device, save_dir):
    probs_list = []
    decoded_words_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            probs, decoded_words = decode(batch, model, ctc_loss, device)
            # change tensor to list
            probs_list.extend(probs.tolist())
            decoded_words_list.extend(decoded_words)
    result = list(zip(probs_list, decoded_words_list))
    with open(os.path.join(save_dir, 'test_predictions.json'), 'w') as f:
        json.dump(result, f, indent=4)

def compute_accuracy(test_dataloader, model, ctc_loss, device):
    # === write your code here ===
    model.eval()
    correct = 0
    total = 0
    for batch in test_dataloader:
        gold_word, _, _, _ = batch
        probs, decoded_words = decode(batch, model, ctc_loss, device)
        for i in range(len(decoded_words)):
            gold_word_string = word_tensor_to_string(gold_word[i])
            decoded_word_string = decoded_words[i]
            if gold_word_string == decoded_word_string:
                correct += 1
            total += 1
    return correct / total


def word_tensor_to_string(word_tensor):
    characters = []
    for i in range(1, len(word_tensor)):
        if word_tensor[i] == 26:
            break
        characters.append(chr(word_tensor[i] + ord('a')))

    return ''.join(characters)

def draw_and_save_loss_graph(train_losses, dev_losses, title, save_dir):
    plt.plot(train_losses, label='train loss')
    plt.plot(dev_losses, label='dev loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.close()
def draw_and_save_accuracy_graph(train_accuracies, dev_accuracies, title, save_dir):
    plt.plot(train_accuracies, label='train accuracy')
    plt.plot(dev_accuracies, label='dev accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.close()

def main(args):
    feature_type = args.feature_type
    system = args.system
    assert feature_type in ['discrete', 'mfcc']
    assert system in ['vanila', 'contrastive']
    # if path does not exist, create it
    if not os.path.exists(os.path.join('results', f'{feature_type}_{system}')):
        os.makedirs(os.path.join('results', f'{feature_type}_{system}'))
    save_dir = os.path.join('results', f'{feature_type}_{system}')

    if feature_type == 'mfcc':
        input_size = 40
    else:
        input_size = 256

    total_set = AsrDataset("data/clsp.trnscr", feature_type, "data/clsp.trnlbls", "data/clsp.lblnames", "data/clsp.trnwav", "data/waveforms/")
    train_set, dev_set = torch.utils.data.random_split(total_set, [0.8, 0.2])

    test_set = AsrDataset("data/clsp.trnscr", feature_type, "data/clsp.devlbls", "data/clsp.lblnames", 
                          "data/clsp.devwav", "data/waveforms/", goldless=True)
    
    tot_dataloader = DataLoader(total_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = LSTM_ASR(feature_type=feature_type, input_size=input_size)
    model.to(device)
    print('model:')
    print(model)
    # your can simply import ctc_loss from torch.nn
    loss_function = torch.nn.CTCLoss(blank=27)
    loss_function_eval = torch.nn.CTCLoss(blank=27, reduction='none')

    # optimizer is provided
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)


    # Containers
    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []
    prev_dev_accuracy = 0
    max_patience = 5
    patience_count = 0
    break_epoch = 0
    # Training
    num_epochs = 100
    for epoch in range(num_epochs):
        loss = train(train_dataloader, model, loss_function, optimizer, device)
        
        dev_loss = get_dev_loss(dev_dataloader, model, loss_function, device)
        accuracy = compute_accuracy(train_dataloader, model, loss_function_eval, device)
        dev_accuracy = compute_accuracy(dev_dataloader, model, loss_function_eval, device)
        print(f'Epoch {epoch+1} train loss: {loss} dev loss: {dev_loss} train accuracy: {accuracy} dev accuracy: {dev_accuracy}')

        train_losses.append(loss)
        dev_losses.append(dev_loss)
        train_accuracies.append(accuracy)
        dev_accuracies.append(dev_accuracy)
        if system == 'contrastive':
            if prev_dev_accuracy >= dev_accuracy:
                patience_count += 1
            else:
                patience_count = 0

            if patience_count >= max_patience:
                break_epoch = epoch
                break
        if dev_accuracy > prev_dev_accuracy:
            prev_dev_accuracy = dev_accuracy
    
    if system == 'contrastive':
        if break_epoch == 0:
            break_epoch = num_epochs
        for epoch in range(break_epoch):
            loss = train(tot_dataloader, model, loss_function, optimizer, device)
            accuracy = compute_accuracy(tot_dataloader, model, loss_function_eval, device)
            train_losses.append(loss)
            train_accuracies.append(accuracy)
            print(f'Epoch {epoch+1} train loss: {loss} train accuracy: {accuracy}')

    draw_and_save_loss_graph(train_losses, dev_losses, f'{system}_{feature_type} loss', save_dir)
    draw_and_save_accuracy_graph(train_accuracies, dev_accuracies, f'{system}_{feature_type} accuracy', save_dir)

    predict_with_probs(test_dataloader, model, loss_function_eval, device, save_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type', type=str, default='discrete')
    parser.add_argument('--system', type=str, default='vanila')
    args = parser.parse_args()
    main(args)
