import torch.nn as nn
import Tool
import torch
from Decoder import DecoderAtten
import random

def make_pair():
    chi = []
    eng = []
    out = []
    with open('after_deal_Chi.txt','r',encoding='utf8') as file0:
        for line in file0:
            chi.append(line.replace("\n",""))
    with open('after_deal_Eng.txt','r',encoding='utf8') as file1:
        for line in file1:
            eng.append(line.replace("\n",""))  
    file0.close()
    file1.close()
    all = []
    for t in range(len(chi)):
        temp = [chi[t],eng[t]]
        all.append(temp)
    return chi,eng

class Encoder(nn.Module):
    ''' 
    dic_size 输入的词典的长度
    hidden_size 隐藏层维度(词向量长度)以及输出维度
    layer_num 网络层数

    output size:[sqe_len,batch_size,hidden_size] 表示了中文句子的整个语义（词向量表示）

    hidden size:[layer_num,batch_size,hidden_size] 中文句子的隐藏语义

    声明网络时，不用给出时间步数，在后续调用的时候会自我迭代

    '''
    def __init__(self, dic_size, hidden_size, layer_num=1, dropout_p=0.1, bidir=False):
        super(Encoder, self).__init__()
        self.input_size = dic_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.dropout_p = dropout_p
        self.bidir = bidir
        # 获得词向量
        self.embedding = nn.Embedding(dic_size, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, layer_num, 
                          dropout=dropout_p, bidirectional=bidir)
        # self.lstm1 = nn.LSTM(hidden_size, hidden_size, layer_num, 
        #                   dropout=dropout_p, bidirectional=bidir)
    
    def forward(self, input_data, input_size, hidden=None):
        # with torch.no_grad():
        embedded = self.embedding(input_data)

        packed1 = nn.utils.rnn.pack_padded_sequence(embedded, input_size)
        out_put, hidden = self.lstm1(packed1, hidden)
        out_put, output_len = nn.utils.rnn.pad_packed_sequence(out_put)  

        packed2 = nn.utils.rnn.pack_padded_sequence(torch.flip(embedded,(2,)), input_size)
        out_put2, _ = self.lstm1(packed2, hidden)
        out_put2, _ = nn.utils.rnn.pad_packed_sequence(out_put2)  
        
        # 双向lstm
        if self.bidir is True:
            out_put = out_put[:, :, :self.hidden_size] + out_put2[:, :, self.hidden_size:]
        return out_put, hidden
        