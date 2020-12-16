'''
IMPORTING THE DATA FROM REPOSITORY
How is its structure?
'''

'''
UNLABELED DATA
'''

import os
import shutil
import numpy as np

basepath = 'C:/Users/Guillem/Desktop/HACKATHON 2020/'
unlabeled_path = basepath + 'Unlabeled audio/'

# Check all directories of Unlabeled data

os.listdir(unlabeled_path)

# Let's create a TRAIN and TEST directory.

train_path = unlabeled_path + 'TRAIN/'
test_path = unlabeled_path + 'TEST/'

if not os.path.exists(train_path):
    os.makedirs(train_path)
    os.makedirs(train_path + 'Cough/')
    os.makedirs(train_path + 'No_Cough/')

if not os.path.exists(test_path):
    os.makedirs(test_path)
    os.makedirs(test_path + 'Cough/')
    os.makedirs(test_path + 'No_Cough/')

# The partition of the data is defined as 50%
# We have 512 cough audios
len(os.listdir(unlabeled_path + 'Cough'))
cough_path = unlabeled_path + 'Cough/'

# We have 431 no cough audios
len(os.listdir(unlabeled_path + 'No_Cough'))
nocough_path = unlabeled_path + 'No_Cough/'

max_len = len(os.listdir(cough_path))//2
for i in os.listdir(cough_path):

    len_train = len(os.listdir(train_path + 'Cough/'))
    old_path = cough_path + i

    if (len_train >= max_len):
        new_path = test_path + 'Cough/' + i
        shutil.move(old_path, new_path)
    else:
        new_path = train_path + 'Cough/' + i
        shutil.move(old_path, new_path)

    if len(os.listdir(cough_path)) == 0:
        os.rmdir(cough_path)


max_len = len(os.listdir(nocough_path))//2
for i in os.listdir(nocough_path):

    len_train = len(os.listdir(train_path + 'No_Cough/'))
    old_path = nocough_path + i

    if (len_train >= max_len):
        new_path = test_path + 'No_Cough/' + i
        shutil.move(old_path, new_path)
    else:
        new_path = train_path + 'No_Cough/' + i
        shutil.move(old_path, new_path)

    if len(os.listdir(nocough_path)) == 0:
        os.rmdir(nocough_path)



'''
LABELED DATA
'''

labeled_path = basepath + 'Labeled audio/'
pos_path = labeled_path + 'Pos/'
neg_path = labeled_path + 'Neg/'
pos_asymp_path = labeled_path + 'Pos_asymp/'

'''
We can extract both cough audios recordings from each participant in a single directory.
'''

positives_path = labeled_path + 'Positives_audios/'
if not os.path.exists(positives_path):
    os.makedirs(positives_path)

for i in os.listdir(pos_path):

    participant_path = pos_path + i

    if ('cough-heavy.wav' in os.listdir(participant_path)):
        old_path = participant_path + '/cough-heavy.wav'
        new_path = positives_path + i + '_cough-heavy.wav'
        shutil.copy(old_path, new_path)

    if ('cough-shallow.wav' in os.listdir(participant_path)):
        old_path = participant_path + '/cough-shallow.wav'
        new_path = positives_path + i + '_cough-shallow.wav'
        shutil.copy(old_path, new_path)
