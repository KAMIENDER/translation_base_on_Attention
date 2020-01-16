from nltk.translate.bleu_score import sentence_bleu

target = []
with open('after_deal_Eng_test.txt','r',encoding='utf8') as file0:
    for line in file0:
        line = line[:-1]
        line  = line.split(' ')
        target.append([line])
file0.close()

pre = []
with open('score_out_test_0.3.txt','r',encoding='utf8') as file1:
    for line in file1:
        line = line[:-1]
        line = line.split(' ')
        pre.append(line)
file1.close()

score = []
for t in range(len(target)):
    scoret = sentence_bleu(target[t],pre[t])
    score.append(scoret)

with open('score_test_0.3.txt','w+',encoding='utf8') as file2:
    for s in score:
        file2.write(str(s)+'\n')
file2.close()
