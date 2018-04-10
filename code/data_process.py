import os,shutil

ROOT_PATH = '/home/liuzheng/project/wrinkling_iron_detection/data/'
TRAIN_RATE = 0.8

PATH = ROOT_PATH + "processed_data/"
ORG_NEG_PATH = ROOT_PATH + 'negtive/'
ORG_POS_PATH = ROOT_PATH + 'positive/'

if( not os.path.exists(PATH + 'test')):
    os.makedirs(PATH + 'test')
if( not os.path.exists(PATH + 'train')):
    os.makedirs(PATH + 'train')


neg_list = os.listdir(ORG_NEG_PATH) 
for filename in neg_list[:int(TRAIN_RATE * len(neg_list))]:
    filedir = ORG_NEG_PATH + filename
    targetdir = ROOT_PATH + 'processed_data/train/neg_' + filename
    shutil.copy(filedir, targetdir)
for filename in neg_list[int(TRAIN_RATE * len(neg_list)):]:
    filedir = ORG_NEG_PATH + filename
    targetdir = ROOT_PATH + 'processed_data/test/neg_' + filename
    shutil.copy(filedir, targetdir)

pos_list = os.listdir(ORG_POS_PATH) 
for filename in pos_list[:int(TRAIN_RATE * len(pos_list))]:
    filedir = ORG_POS_PATH + filename
    targetdir = ROOT_PATH + 'processed_data/train/pos_' + filename
    shutil.copy(filedir, targetdir)
for filename in pos_list[int(TRAIN_RATE * len(pos_list)):]:
    filedir = ORG_POS_PATH + filename
    targetdir = ROOT_PATH + 'processed_data/test/pos_' + filename
    shutil.copy(filedir, targetdir)



      
