import string
import torch
import librosa
import os
import pandas as pd
import random
from torch.utils.data import Dataset

class AsrDataset(Dataset):
    def __init__(self, scr_file, feature_type='discrete', feature_file=None,
                 feature_label_file=None,
                 wav_scp=None, wav_dir=None, goldless=False):
        """
        :param scr_file: clsp.trnscr
        :param feature_type: "quantized" or "mfcc"
        :param feature_file: clsp.trainlbls or clsp.devlbls
        :param feature_label_file: clsp.lblnames
        :param wav_scp: clsp.trnwav or clsp.devwav
        :param wav_dir: wavforms/

        Set so that this dataset already tokenizes input, output.
        """
        assert feature_type in ['discrete', 'mfcc']


        if feature_type == 'discrete':

            with open(feature_file, 'r') as f:
                features = [line.strip().split(" ") for i, line in enumerate(f) if i > 0]

            with open(feature_label_file, 'r') as f:
                feature_labels = [line.strip() for i, line in enumerate(f) if i > 0]
            features = convert_features_to_index(features, feature_labels)
        
        else:
            features = self.compute_mfcc(wav_scp, wav_dir)

        if goldless:
            feature_count = len(features)
            # create dummy words
            words = [list('abcd') for _ in range(feature_count)]
            tokenized_words = [torch.tensor([26] + [ord(c) - ord('a') for c in word] + [26]) for word in words]



        else:
            with open(scr_file, 'r') as f:
                words = [list(line.strip()) for i, line in enumerate(f) if i > 0]
            tokenized_words = [torch.tensor([26] + [ord(c) - ord('a') for c in word] + [26]) for word in words]
            


        self.script = list(zip(tokenized_words, features))
            


    def __len__(self):
        """
        :return: num_of_samples
        """
        return len(self.script)

    def __getitem__(self, idx):
        """
        Get one sample each time. Do not forget the leading- and trailing-silence.
        :param idx: index of sample
        :return: spelling_of_word, feature
        """
        # === write your code here ===
        return self.script[idx]


    # This function is provided
    def compute_mfcc(self, wav_scp, wav_dir):
        """
        Compute MFCC acoustic features (dim=40) for each wav file.
        :param wav_scp:
        :param wav_dir:
        :return: features: List[np.ndarray, ...]
        """
        features = []
        with open(wav_scp, 'r') as f:
            for wavfile in f:
                wavfile = wavfile.strip()
                if wavfile == 'jhucsp.trnwav' or wavfile == "jhucsp.devwav":  # skip header
                    continue
                wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
                feats = librosa.feature.mfcc(y=wav, sr=16e3, n_mfcc=40, hop_length=160, win_length=400).transpose()
                features.append(feats)
        features = [torch.tensor(feature) for feature in features]
        return features


def convert_features_to_index(features, feature_labels):
    new_out = []
    for i, v in enumerate(features):
        n = [feature_labels.index(f) for f in v]
        new_out.append(torch.tensor(n))
    return new_out

def character_to_index(character):
    return ord(character) - ord('a')

if __name__ == "__main__":
    ds = AsrDataset("data/clsp.trnscr", "mfcc", "data/clsp.trnlbls", "data/clsp.lblnames", "data/clsp.trnwav", "data/waveforms/")