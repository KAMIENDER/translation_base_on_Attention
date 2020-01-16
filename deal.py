import re
import jieba

pattern = ["(",")","[","]","?",".",",","!",'“','”']

# mode 0: English
def pre_process(path,mode,outpath):
    with open(outpath,'w+',encoding='utf8') as file1:
        with open(path,'r',encoding='utf8') as file0:
            for line in file0:
                if mode == 0:
                    line = line.lower().strip()
                    for pa in pattern:
                        line = line.replace(pa," "+pa+" ")
                    line = line.replace("  "," ")
                    line = line.rstrip().strip()

                else:
                    line.strip()
                    temp = jieba.cut(line)
                    line = ""
                    for word in temp:
                        if word == '\n' or word == ' ' :
                            t = 1
                            continue
                        line = line + word + ' '
                file1.write(line+'\n')
        file0.close()
    file1.close()


if __name__ == "__main__":
    pre_process("final_project_dataset/dataset_10000/dev_target_1000.txt",0,'after_deal_Eng_dev.txt')
