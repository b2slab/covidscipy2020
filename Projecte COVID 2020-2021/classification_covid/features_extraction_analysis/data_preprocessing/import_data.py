import os
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

basepath = 'C:/Users/Guillem/Desktop/New Coswara data/prueba/'

metadata = pd.read_csv('C:/Users/Guillem/Desktop/New Coswara data/20201221.csv')
metadata['covid_status'].unique()

labels = metadata['covid_status'].map(lambda label: 'pos' if ('positive' in label) else 'neg')
id = metadata.id

df = pd.concat([id,labels], axis = 1)


pos_path = basepath + 'pos/'
neg_path = basepath + 'neg/'

if not os.path.exists(pos_path):
    os.makedirs(pos_path)
    if not os.path.exists(neg_path):
        os.makedirs(neg_path)

basepath = 'C:/Users/Guillem/Desktop/New Coswara data/prueba/20201221/'
for patient_path in os.listdir(basepath):
    label = df['covid_status'][df['id'] == patient_path].item()

    if label == 'pos':
        shutil.move(os.path.join(basepath, patient_path), os.path.join(pos_path, patient_path))
    elif label == 'neg':
        shutil.move(os.path.join(basepath, patient_path), os.path.join(neg_path, patient_path))


for patient_folder in os.listdir(pos_path):
    for audio in os.listdir(os.path.join(pos_path,patient_folder)):
        if audio == 'cough-heavy.wav':
            old_path = os.path.join(os.path.join(pos_path, patient_folder), 'cough-heavy.wav')
            new_path = os.path.join(pos_path, '{}_cough-heavy.wav'.format(patient_folder))
            shutil.copy(old_path, new_path)

        if audio == 'cough-shallow.wav':
            old_path = os.path.join(os.path.join(pos_path, patient_folder), 'cough-shallow.wav')
            new_path = os.path.join(pos_path, '{}_cough-shallow.wav'.format(patient_folder))
            shutil.copy(old_path, new_path)

for patient_folder in os.listdir(neg_path):
    for audio in os.listdir(os.path.join(neg_path,patient_folder)):
        if audio == 'cough-heavy.wav':
            old_path = os.path.join(os.path.join(neg_path, patient_folder), 'cough-heavy.wav')
            new_path = os.path.join(neg_path, '{}_cough-heavy.wav'.format(patient_folder))
            shutil.copy(old_path, new_path)

        if audio == 'cough-shallow.wav':
            old_path = os.path.join(os.path.join(neg_path, patient_folder), 'cough-shallow.wav')
            new_path = os.path.join(neg_path, '{}_cough-shallow.wav'.format(patient_folder))
            shutil.copy(old_path, new_path)

if len(os.listdir(basepath)) == 0:
    os.rmdir(basepath)


### REMOVE MANUALLY FOLDERS
for file in os.listdir(pos_path):
    if not 'wav' in file:
        os.rmdir(os.path.join(pos_path, file))





#### Let's extract the spectrogram from the wav files

from audio_data_augmentation import *

spectrograms_pos_path = pos_path + 'spectrograms/'
if not os.path.exists(spectrograms_pos_path):
    os.makedirs(spectrograms_pos_path)

onlyfiles_pos = [os.path.join(pos_path, f) for f in os.listdir(pos_path) if os.path.isfile(os.path.join(pos_path, f))]

for audio_path in onlyfiles_pos:
    wav_to_spectrogram(audio_path, spectrograms_pos_path, augment_data=False, height=448, width=448)



spectrograms_negatives_path = neg_path + 'spectrograms/'
if not os.path.exists(spectrograms_negatives_path):
    os.makedirs(spectrograms_negatives_path)


onlyfiles_neg = [os.path.join(neg_path, f) for f in os.listdir(neg_path) if os.path.isfile(os.path.join(neg_path, f))]

for audio_path in onlyfiles_neg:
    wav_to_spectrogram(audio_path, spectrograms_negatives_path, height=448, width=448)



### MANUALLY MOVE FILES TO A SINGLE FOLDER


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

img_height, img_width, channels = 447, 447, 3
IMG_SIZE = (img_height, img_width)

newdata_path = 'C:/Users/Guillem/Desktop/New Coswara data/spectrograms/'
newdata_dataset = image_dataset_from_directory(newdata_path,image_size=IMG_SIZE)


model = tf.keras.models.load_model(os.path.join(newdata_path, 'ResNet152V2_model_covid_no_augmentation.h5'))

loss, accuracy = model.evaluate(newdata_dataset, batch_size=32)
print('New data accuracy :', accuracy)

'''
labels = tf.concat([y for x, y in newdata_dataset], axis=0).numpy()
predictions = tf.concat([tf.nn.sigmoid(model.predict_on_batch(x).flatten()).numpy() for x, y in newdata_dataset], axis = 0).numpy()

'''
labels = []
predictions = []

for img, label in newdata_dataset:
    _predictions = model.predict_on_batch(img).flatten()
    _predictions = tf.nn.sigmoid(_predictions).numpy()

    for lab in label.numpy():
      labels.append(lab)

    for pred in _predictions:
      predictions.append(pred)

predictions = np.array(predictions)
labels = np.array(labels)



y_true, y_predicted = Confusion_Matrix(y_true = labels,
                                       y_predicted = predictions,
                                       pred_prob = True)




'''
Vamos a comprobar la eficacia del modelo del HACKATHON
en nuevos datos nunca vistos
'''
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchvision
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import librosa

basepath = 'C:/Users/Guillem/Desktop/New Coswara data/prueba/'
pos_path = basepath + 'pos/'
neg_path = basepath + 'neg/'

# Let's start with positives
files_pos = os.listdir(pos_path)

predictions = []
for file in files_pos:

    file_path = os.path.join(pos_path, file)

    waveform, sample_rate = torchaudio.load(file_path)
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    specgram_resize = torchvision.transforms.Resize((448,448))(specgram)
    plt.figure(frameon=False)
    plt.axis('off')
    specgram_resize += torch.ones(list(specgram_resize.shape))*1e-12
    plt.imshow(specgram_resize.log2()[0,:,:].numpy(), cmap='gray')
    plt.savefig(file_path.strip('.wav')+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # image = Image.open(file_path.strip('.wav')+'.png').convert('RGB')
    image = Image.open(file_path.strip('.wav')+'.png').convert('RGB')
    loader = torchvision.transforms.Compose([
            #torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((112,112)),
            torchvision.transforms.ToTensor(),
    ])
    image = loader(image)
    image.unsqueeze_(0)

    model = torch.load(basepath + 'fine_tuned_transfer_augmented.pt')
    model.eval()
    out = model(image)

    predictions.append(out)

files_neg = os.listdir(neg_path)

pred_neg = []
for file in files_neg:

    file_path = os.path.join(neg_path, file)

    waveform, sample_rate = torchaudio.load(file_path)
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    specgram_resize = torchvision.transforms.Resize((224,224))(specgram)
    plt.figure(frameon=False)
    plt.axis('off')
    specgram_resize += torch.ones(list(specgram_resize.shape))*1e-12
    plt.imshow(specgram_resize.log2()[0,:,:].numpy(), cmap='gray')
    plt.savefig(file_path.strip('.wav')+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    image = Image.open(file_path.strip('.wav')+'.png').convert('RGB')
    loader = torchvision.transforms.Compose([
            #torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((112,112)),
            torchvision.transforms.ToTensor(),
    ])
    image = loader(image)
    image.unsqueeze_(0)

    model = torch.load(basepath + 'fine_tuned_transfer_augmented.pt')
    model.eval()
    out = model(image)

    pred_neg.append(out)



import numpy as np

pos = []
for pred in predictions:
    pos.append(pred.detach().numpy()[0][0])

neg = []
for pred in pred_neg:
    neg.append(pred.detach().numpy()[0][0])


y_true = np.append(np.ones(len(pos)), np.zeros(len(neg)))
y_pred = np.append(pos, neg)

_y_pred = pd.Series(y_pred).map(lambda x: 0 if x > 0.8 else 1)

Confusion_Matrix(y_true, y_pred, pred_prob = True)








def Confusion_Matrix(y_true, y_predicted, binarized_true = False, binarized_pred = False, pred_prob = False):

    '''
    Generate a confusion matrix for binary classification. Additionally plots the ROC and AUC if the
    output of the model predicts probabilities
    @params:
        y_true          - A list of integers or strings for known classes
        y_predicted     - A list of integers, strings or probabilities for predicted classes
        binaried_true   - If the y_true are strings (FALSE,TRUE) converts to numerical [0,1]
        binaried_pred   - If the y_predicted are strings (FALSE,TRUE) converts to numerical [0,1]
        pred_prob       - If the predictions of the model are probabilities, then an optimal threshold can be compute
                          as well as plotting the ROC and AUC

    @return:
        confusion matrix
        classification Report
        optimal threshold
        AUC
        Plot of ROC
        y_true, y_predicted   (treated)

    @ Precision: What proportion of positive identifications was actually correct?
                 Our model has a precision of 0.5—in other words, when it predicts a recording is a cough, it is correct 50% of the time.

    @ Recall: What proportion of actual positives was identified correctly?
              Our model has a recall of 0.11—in other words, it correctly identifies 11% of all cough recordings.
    '''

    y_true = pd.Series(y_true, name = 'Actual')
    y_predicted = pd.Series(y_predicted, name='Predicted')

    if (binarized_true):
        y_true = y_true.map({True:1, False:0})

    if (binarized_pred):
        y_predicted = y_predicted.map({True:1, False:0})

    if (pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_predicted, pos_label=1)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print("\n __________ \n")
        print("Optimal Threshold: {} \n __________ \n".format(round(optimal_threshold,4)))

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC curve (area = {})'.format(round(roc_auc,3)))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        y_predicted = pd.Series(np.where(y_predicted>=optimal_threshold, 1, 0), name='Predicted')


    df_confusion = pd.crosstab(y_true, y_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print("\n __________ \n")
    print("Confusion Matrix: \n __________ \n")
    print(df_confusion)
    print("\n __________ \n")
    print("Classification Report: \n __________ \n")
    print(classification_report(y_true, y_predicted))

    return y_true, y_predicted
