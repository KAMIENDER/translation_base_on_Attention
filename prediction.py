import model
import Tool
import Decoder
import torch.nn as nn
import torch
import torch.optim as optim
import random
import train

def pred():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 100
    hidden_size = 32
    layer_num = 2
    learn_rate = 0.001
    decoder_ratio = 0.8
    teacher_ratio = 0

    teacher_ratio = 0.7
    batch_size = 20


    

    _,chi,eng = Tool.read_data('after_deal_Chi.txt','after_deal_Eng.txt')

    encoder = model.Encoder(chi.word_num,hidden_size,layer_num).to(device)
    encoder.load_state_dict(torch.load('encode2.m', map_location=device))
    decoder = Decoder.DecoderAtten(batch_size,hidden_size,eng.word_num,layer_num=layer_num).to(device)
    decoder.load_state_dict(torch.load('decoder2.m', map_location=device))

    pair,_,_ = Tool.read_data("after_deal_Chi.txt",'after_deal_Eng.txt')

    

    all_out = []

    with open('score_out_dev.txt','w+',encoding='utf8') as file2:
        for t in range(1000//batch_size):
            input_data, input_len, output_data, output_len = Tool.get_batch(batch_size,pair,chi,eng,t)
            pre1,score1 = train.pre_detail(input_data, input_len, output_data, output_len, encoder,  decoder, batch_size, hidden_size, device,1)
            pre0,score0 = train.pre_detail(input_data, input_len, output_data, output_len, encoder,  decoder, batch_size, hidden_size, device,0)

            pre0 = pre0.transpose(0,1)
            pre1 = pre1.transpose(0,1)
            score1 = score1.transpose(0,1)
            score0 = score0.transpose(0,1)
            pre1 = pre1.data.cpu().tolist()
            pre0 = pre0.data.cpu().tolist()

            for t in range(len(pre1)):
                if score1[t].sum() > score0[t].sum():
                    file2.write(eng.translate(pre1[t])+'\n')
                else:
                    file2.write(eng.translate(pre0[t])+'\n')
        
            
    file2.close()
    

if __name__ == "__main__":
    with torch.no_grad():
        pred()
    
    