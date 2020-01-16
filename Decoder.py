import torch.nn as nn
import Tool
import torch
import torch.nn.functional as F
import time

def my_log_softmax(x):
    size = x.size()
    res = F.log_softmax(x,dim=2)
    res = res.view(size[0], size[1], -1)
    return res

class Atten(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(Atten, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.atten = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, rnn_outputs, encoder_outputs):       
        rnn_outputs = rnn_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        encoder_outputs = self.atten(encoder_outputs).transpose(1, 2)
        atten_energies = torch.exp(rnn_outputs.bmm(encoder_outputs))
        res = my_log_softmax(atten_energies)
        return res

class DecoderAtten(nn.Module):
    def __init__(self,batch_size,hidden_size, output_size, layer_num=1, dropout_p=0.1):
        super(DecoderAtten, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, layer_num, dropout=dropout_p,bidirectional=False)
        self.atten = Atten(hidden_size, batch_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, last_hidden, encoder_output):
        
        # with torch.no_grad():
        batch_size = input_seq.size()[1]
        tar_len = input_seq.size()[0]
        ins = encoder_output.size()[0]
        
        embedded = self.embedding(input_seq)
        embedded = embedded.view(tar_len, batch_size, self.hidden_size)

        rnn_output, hidden = self.lstm(embedded, last_hidden)

        atten_weights = self.atten(rnn_output, encoder_output)

        context = atten_weights.bmm(encoder_output.transpose(0, 1))
        context = context.transpose(0, 1)
        
        output_context = torch.cat((rnn_output, context), 2)
        output_context = self.concat(output_context)
        concat_output = torch.tanh(output_context)
        
        output = self.out(concat_output)
        output = my_log_softmax(output)
        return output, hidden, atten_weights

    def init_outputs(self, seq_len, batch_size):
        outputs = torch.zeros(seq_len, batch_size, self.output_size)
        return Tool.get_variable(outputs)
    
    def create_input_seqs(self, seq_len, batch_size):
        sos = [Tool.SOS] * batch_size
        sos = [sos] * seq_len
        return torch.LongTensor(sos)
