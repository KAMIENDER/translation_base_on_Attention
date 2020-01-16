
import random
import torch

PAD = 0
SOS = 1
EOS = 2

class Tool(object):
    def __init__(self, name):
        self.name = name
        self.init_params()

    def init_params(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS"}
        self.word_num = 3

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.word2count[word] = 1
            self.index2word[len(self.index2word)] = word
            self.word_num += 1
        else:
            self.word2count[word] += 1
    
    def add_sentence_word(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)
    
    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def translate(self,li):
        out = ""
        for num in li:
            if num != EOS:
                out = out + self.index2word[num] + " "
            else:
                break
        return out

    
def read_file(path):
    lines = []
    with open(path, 'r',encoding='utf8') as file0:
        num = 0
        for line in file0:
            line = line[:-1]
            lines.append(line)
            num = num + 1
    file0.close()
    return lines


def read_data(Chi_file, Eng_file):
    chi_lang = Tool('Chinese')
    target_lang = Tool('English')
    input_lines = read_file(Chi_file)
    target_lines = read_file(Eng_file)
    n_lines = len(input_lines)
    pairs = []
    for i in range(n_lines):
        input_line = input_lines[i]
        target_line = target_lines[i]
        chi_lang.add_sentence_word(input_line)
        target_lang.add_sentence_word(target_line)
        pairs.append([input_line, target_line])
    return pairs, chi_lang, target_lang,

def indexes_from_sentence(tool, sentence):
    out = []
    for word in sentence.split(' '):
        if tool.word2index.get(word)!=None:
            out.append(tool.word2index[word])
    out.append(EOS)
    return out

def pad_seq(seq, length):
    seq += [PAD for i in range(length - len(seq))]
    return seq




def random_batch(batch_size, pairs, input_lang, target_lang):
    input_seqs = []
    target_seqs = []

    seq_pairs = sorted(zip(input_seqs, target_seqs), key = lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    for i in range(batch_size):
        p = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, p[0]))
        target_seqs.append(indexes_from_sentence(target_lang, p[1]))
    
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = torch.LongTensor(input_padded).transpose(0, 1)
    target_var = torch.LongTensor(target_padded).transpose(0, 1)
    return input_var, input_lengths, target_var, target_lengths

def get_batch(batch_size, pairs, input_lang, target_lang,t):
    input_seqs = []
    target_seqs = []
    
    for i in range(batch_size):
        p = pairs[t*batch_size+i]
        input_seqs.append(indexes_from_sentence(input_lang, p[0]))
        target_seqs.append(indexes_from_sentence(target_lang, p[1]))
    
    seq_pairs = sorted(zip(input_seqs, target_seqs), key = lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = torch.LongTensor(input_padded).transpose(0, 1)
    target_var = torch.LongTensor(target_padded).transpose(0, 1)
    return input_var, input_lengths, target_var, target_lengths