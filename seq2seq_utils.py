import codecs
from tqdm import tqdm_notebook as tqdm
from collections import Counter, namedtuple
from torch.utils.data import Dataset, DataLoader
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from torch.utils.data import Dataset, DataLoader

## datasets

PAD = 0
BOS = 1
EOS = 2
UNK = 3

MAX_SRC_DOC_LENGTH = 400
MAX_TGT_DOC_LENGTH = 100
MAX_DOC_LENGTH = 200

class AttrDict(dict):
    """ Access dictionary keys like attribute 
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self
        
class MLPDataset(Dataset):
    def __init__(self, src_path, tgt_path, max_vocab_size=50000, MAX_SRC_DOC_LENGTH=400):
        """ Note: If src_vocab, tgt_vocab is not given, it will build both vocabs.
            Args: 
            - src_path, tgt_path: text file with tokenized sentences.
            - src_vocab, tgt_vocab: data structure is same as self.build_vocab().
        """
        print('='*100)
        print('Dataset preprocessing log:')
        self.MAX_SRC_DOC_LENGTH=MAX_SRC_DOC_LENGTH
        print('- Loading and tokenizing source sentences...')
        self.src_sents = self.load_sents(src_path, sent_type = 'src')
        print('- Loading targets')
        
        with open(tgt_path, 'rb') as fp:
            data = pickle.load(fp)
    
        self.tgts = [item[1] for item in data]
        self.scts = [item[0] for item in data]
                
        self.src_counter = self.build_counter(self.src_sents)
                
        self.src_vocab = self.build_vocab(self.src_counter, max_vocab_size)
        
    def __len__(self):
        return len(self.src_sents)
    
    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgts = self.tgts[index]
        src_seq = self.tokens2ids(src_sent, self.src_vocab.token2id, append_BOS=False, append_EOS=True)
        scts = self.scts[index]

        return src_sent, src_seq, tgts, scts
        
    def load_sents(self, file_path, sent_type):
        sents = []
        with codecs.open(file_path) as file:
            for sent in tqdm(file.readlines()):
                tokens = [token for token in sent.split()]
                if sent_type == 'src':
                    tokens_trunc = tokens[:self.MAX_SRC_DOC_LENGTH]
                elif sent_type == 'tgt':
                    tokens_trunc = tokens[:MAX_TGT_DOC_LENGTH]
                else:
                    print("FYI: no sentence type")
                    tokens_trunc = tokens[:MAX_DOC_LENGTH]
                sents.append(tokens_trunc)
        return sents
    
    def build_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter.update(sent)
        return counter
    
    def build_vocab(self, counter, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
        vocab.token2id.update({token: _id+4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v:k for k,v in tqdm(vocab.token2id.items())}    
        return vocab
    
    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq

class NMTDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab=None, tgt_vocab=None, max_vocab_size=50000, share_vocab=True):
        """ Note: If src_vocab, tgt_vocab is not given, it will build both vocabs.
            Args: 
            - src_path, tgt_path: text file with tokenized sentences.
            - src_vocab, tgt_vocab: data structure is same as self.build_vocab().
        """
        print('='*100)
        print('Dataset preprocessing log:')
        
        print('- Loading and tokenizing source sentences...')
        self.src_sents = self.load_sents(src_path, sent_type = 'src')
        print('- Loading and tokenizing target sentences...')
        self.tgt_sents = self.load_sents(tgt_path, sent_type = 'tgt')
        
        if src_vocab is None or tgt_vocab is None:
            print('- Building source counter...')
            self.src_counter = self.build_counter(self.src_sents)
            print('- Building target counter...')
            self.tgt_counter = self.build_counter(self.tgt_sents)

            if share_vocab:
                print('- Building source vocabulary...')
                self.src_vocab = self.build_vocab(self.src_counter + self.tgt_counter, max_vocab_size)
                print('- Building target vocabulary...')
                self.tgt_vocab = self.src_vocab
            else:
                print('- Building source vocabulary...')
                self.src_vocab = self.build_vocab(self.src_counter, max_vocab_size)
                print('- Building target vocabulary...')
                self.tgt_vocab = self.build_vocab(self.tgt_counter, max_vocab_size)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            share_vocab = src_vocab == tgt_vocab
                        
        print('='*100)
        print('Dataset Info:')
        print('- Number of source sentences: {}'.format(len(self.src_sents)))
        print('- Number of target sentences: {}'.format(len(self.tgt_sents)))
        print('- Source vocabulary size: {}'.format(len(self.src_vocab.token2id)))
        print('- Target vocabulary size: {}'.format(len(self.tgt_vocab.token2id)))
        print('- Shared vocabulary: {}'.format(share_vocab))
        print('='*100 + '\n')
    
    def __len__(self):
        return len(self.src_sents)
    
    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]
        src_seq = self.tokens2ids(src_sent, self.src_vocab.token2id, append_BOS=False, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.tgt_vocab.token2id, append_BOS=False, append_EOS=True)

        return src_sent, tgt_sent, src_seq, tgt_seq
    
    def load_sents(self, file_path, sent_type):
        sents = []
        with codecs.open(file_path) as file:
            for sent in tqdm(file.readlines()):
                tokens = [token for token in sent.split()]
                if sent_type == 'src':
                    tokens_trunc = tokens[:MAX_SRC_DOC_LENGTH]
                elif sent_type == 'tgt':
                    tokens_trunc = tokens[:MAX_TGT_DOC_LENGTH]
                else:
                    print("FYI: no sentence type")
                    tokens_trunc = tokens[:MAX_DOC_LENGTH]
                sents.append(tokens_trunc)
        return sents
    
    def build_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter.update(sent)
        return counter
    
    def build_vocab(self, counter, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
        vocab.token2id.update({token: _id+4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v:k for k,v in tqdm(vocab.token2id.items())}    
        return vocab
    
    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq
    
def collate_fn(data):
    """
    Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    
    Args:
        data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
        - src_sents, tgt_sents: batch of original tokenized sentences
        - src_seqs, tgt_seqs: batch of original tokenized sentence ids
    Returns:
        - src_sents, tgt_sents (tuple): batch of original tokenized sentences
        - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
        - src_lens, tgt_lens (tensor): (batch_size)
       
    """
    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, tgt_sents, src_seqs, tgt_seqs = zip(*data)
    
    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)
    
    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0,1)
    tgt_seqs = tgt_seqs.transpose(0,1)

    return src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens

def collate_fn2(data):
    """
    Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    
    Args:
        data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
        - src_sents, tgt_sents: batch of original tokenized sentences
        - src_seqs, tgt_seqs: batch of original tokenized sentence ids
    Returns:
        - src_sents, tgt_sents (tuple): batch of original tokenized sentences
        - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
        - src_lens, tgt_lens (tensor): (batch_size)
       
    """
    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, src_seqs, tgts, scts = zip(*data)
    
    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    
    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0,1)

    return src_sents, src_seqs, tgts, src_lens, scts