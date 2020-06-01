import glob
import csv
import os
txt = glob.glob("/mnt/edisk/backup/dataset/semantic_raw/*.txt")
print(len(txt))
txt_train = txt[0:236]
txt_val = txt[237:241]
txt_test = txt[242:246]

os.chdir("/mnt/edisk/backup/filelists")


with open('FileList_train.txt', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(txt_train)

with open('FileList_val.txt', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(txt_val)

with open('FileList_test.txt', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(txt_test)

labels =glob.glob("/mnt/edisk/backup/dataset/semantic_raw/*.labels")


label_train = labels[0:236]
label_val = labels[237:241]

with open('LabelList_train.txt', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(label_train)

with open('LabelList_val.txt', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(label_val)
