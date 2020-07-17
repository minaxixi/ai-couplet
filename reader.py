"""reader.py

Read couplet data and create the dictionary
"""

import numpy as np

class CoupletReader():

    def __init__(self, input_file, output_file, vocab_file, max_len, max_char):
        '''
        input_file: the first half of the couplet
        output_file: the second half of the couplet
        vocab_file: the file containing all the characters, sorted by reversed frequency
        max_len: max length of the couplet in raw characters (excluding start_token, pad_token, etc.)
        max_char: max number of characters in the vocabulary
        '''

        self.input_file = input_file
        self.output_file = output_file
        self.vocab_file = vocab_file
        self.start_token = '<s>'
        self.max_len = max_len

        self.special_chars = ['<pad>', self.start_token]

        # read couplets
        self.data, self.target = self._read_file()
        data_padded, target_padded = self._pad_sentences(self.data, self.target)

        # read character list and create mapping
        vocab = self._read_vocab()
        char2idx, idx2char = self._create_vocab_mapping(vocab)
        self.char2idx = char2idx
        self.idx2char = idx2char

        # pad and encode the couplets
        vocab, data_padded, target_padded, data_encoded, target_encoded = self._encode_vocab(vocab, data_padded, target_padded, max_char)

        self.data_padded, self.target_padded = data_padded, target_padded
        self.data_encoded, self.target_encoded = data_encoded, target_encoded

        # filter vocab to have at most max_char characters
        vocab = vocab[:max_char]
        self.vocab_size = len(vocab)
        self.idx2char = self.idx2char[:max_char]
        self.char2idx = {k:v for k,v in self.char2idx.items() if v < max_char}

    def encode(self, sentence):
        ''' encode the sentence to integer '''
        return [self.char2idx[char] for char in sentence]

    def decode(self, sentence_int):
        ''' decode the integers to sentence '''
        return [self.idx2char[idx] for idx in sentence_int]

    def _read_file(self):

        start_token = self.start_token

        # read the first half of the couplet
        input_f = open(self.input_file, 'rb')
        data = []
        for input_line in input_f:
            line = input_line.decode('utf-8')[:-1].split()
            data.append(line[:self.max_len])
        input_f.close()

        # read the second half of the couplet
        output_f = open(self.output_file, 'rb')
        target = []
        for output_line in output_f:
            line = output_line.decode('utf-8')[:-1].split()
            target.append([start_token] + line[:self.max_len])
        output_f.close()

        return data, target

    def _pad_sentences(self, data, target):

        data_padded = []
        for sentence in data:
            pad_len = self.max_len - len(sentence)
            data_padded.append(sentence + ['<pad>' for i in range(pad_len)])

        target_padded = []
        for sentence in target:
            pad_len = self.max_len + 1 - len(sentence) # +1 is to account for the start token
            target_padded.append(sentence + ['<pad>' for i in range(pad_len)])

        return data_padded, target_padded

    def _read_vocab(self):

        # the vocab file is sorted by char frequency
        vocab_f = open(self.vocab_file, 'rb')
        vocab = []
        for vocab_line in vocab_f:
            line = vocab_line.decode('utf-8').split()[0]
            vocab.append(line)
        vocab_f.close()

        vocab = self.special_chars + vocab

        return vocab

    def _create_vocab_mapping(self, vocab):
        char2idx = {u:i for i, u in enumerate(vocab)}
        idx2char = np.array(list(vocab))
        return char2idx, idx2char

    def _encode_vocab(self, vocab, data_padded, target_padded, max_char):
        ''' encode the couplets and filter out rare characters'''

        # first encode
        data_encoded = np.array([self.encode(x) for x in data_padded])
        target_encoded = np.array([self.encode(x) for x in target_padded])

        # filter if there are rare characters (index > max_char)
        cond = (np.all(data_encoded < max_char, axis=1)) & (np.all(target_encoded < max_char, axis=1))
        data_encoded, target_encoded = data_encoded[cond], target_encoded[cond]

        # re-compute the padded sentence
        data_padded = [self.decode(sentence) for sentence in data_encoded]
        target_padded = [self.decode(sentence) for sentence in target_encoded]

        return vocab, data_padded, target_padded, data_encoded, target_encoded
