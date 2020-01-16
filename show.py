import numpy as np
import matplotlib.pyplot as plt

score = []

with open('score_test.txt','r',encoding='utf8') as file0:
    for line in file0:
        temp = line[:-1]
        temp = float(temp)
        score.append(temp)

print(sum(score)/len(score))
file0.close()

loss = []
x = np.linspace(0,100,100)
t = 0
all = 0
with open('loss_0.3.txt','r',encoding='utf8') as file1:
    for line in file1:
        line = line.split(' ')
        temp = float(line[4])
        all = all + temp
        t = t + 1
        if t==400:
            loss.append(all/400)
            all=0
            t=0
y = np.array(loss)
plt.xlabel('time')
plt.ylabel('loss')
plt.plot(x,y)
plt.show()