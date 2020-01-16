import model
import Tool
import Decoder
import torch.nn as nn
import torch
import torch.optim as optim
import random

def train_detail(input_batch, input_len, target_batch, target_len,encoder,decoder,bathc_size,hidden_size,teacher_ratio,encoder_optimizer,decoder_optimizer,device):
    # 梯度清零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    max_target_len = max(target_len)

    # 初始输入
    en_out, en_hidden = encoder(input_batch.to(device), input_len)
    de_input = decoder.create_input_seqs(max_target_len, bathc_size)
    temp = en_hidden[0]
    de_out, de_hidden, atten = decoder(de_input.to(device),en_hidden,en_out.to(device))

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    for t in range(max_target_len):

        if random.random() < teacher_ratio:
            de_input = target_batch
        else:
            words = torch.max(de_out,2,True)[1]
            words = words.squeeze(2)
            de_input = words
        
        de_out, de_hidden, atten = decoder(de_input.to(device),de_hidden,en_out.to(device))

    result = de_out.reshape(de_out.shape[0]*de_out.shape[1],de_out.shape[2])
    tar_temp = target_batch.reshape(de_out.shape[0]*de_out.shape[1])

    loss = criterion(result, tar_temp.to(device))
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss

def pre_detail(input_batch, input_len, target_batch, target_len,encoder,decoder,bathc_size,hidden_size,device, dim):
    # 梯度清零

    max_target_len = max(target_len)

    # 初始输入
    en_out, en_hidden = encoder(input_batch.to(device), input_len)
    de_input = decoder.create_input_seqs(max_target_len, bathc_size)
    de_out, de_hidden, atten = decoder(de_input.to(device),en_hidden,en_out.to(device))

    for t in range(max_target_len):

        if t == 0:
            de_input = de_out.sort(-1,descending=True)[1][:,:,dim]
        else:
            de_input = torch.max(de_out,2,True)[1]
            de_input = de_input.squeeze(2)
        
        de_out, de_hidden, atten = decoder(de_input.to(device),de_hidden,en_out.to(device))

    score,words = torch.max(de_out,2,True)
    words = words.squeeze(2)
    score = score.squeeze(2)

    return words,score


def train():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 100
    batch_size = 20
    hidden_size = 32
    layer_num = 2
    learn_rate = 0.001
    decoder_ratio = 0.8

    teacher_ratio = 1

    loss = 0
    total_loss = []

    pair,chi,eng = Tool.read_data('after_deal_Chi.txt','after_deal_Eng.txt')

    encoder = model.Encoder(chi.word_num,hidden_size,layer_num).to(device)
    decoder = Decoder.DecoderAtten(batch_size,hidden_size,eng.word_num,layer_num=layer_num).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learn_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learn_rate * decoder_ratio)

    loss = 0

    with open('loss_0.3.txt','w+',encoding='utf8') as file3:
        for i in range(epoch):
            for t in range(8000//batch_size):
                input_batches, input_lengths, target_batches, target_lengths = Tool.get_batch(batch_size, pair, chi, eng,t)
                loss = train_detail(input_batches,input_lengths,target_batches,target_lengths,encoder,decoder,batch_size,hidden_size,teacher_ratio,encoder_optimizer,decoder_optimizer,device)
            
                print("epoch: ", i, " | loss: ", loss)
                file3.write("epoch: "+str(i)+ " | loss: "+str(loss.data.item())+'\n')

        
            torch.save(encoder.state_dict(), "encode2_1.m")
            torch.save(decoder.state_dict(), "decoder2_1.m")

    file3.close()

if __name__ == "__main__":
    train()