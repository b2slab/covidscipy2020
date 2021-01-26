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
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import pylab
import random

basepath = 'C:/Users/Guillem/Desktop/New Coswara data/prueba/'
pos_path = os.path.join(basepath, 'pos/')
neg_path = os.path.join(basepath, 'neg/')

aug_path = os.path.join(basepath, 'augmented_data/')
if not os.path.exists(aug_path):
    os.makedirs(aug_path)
    os.makedirs(aug_path + 'pos/')
    os.makedirs(aug_path + 'neg/')

'''
Por cada tos que tengamos, haremos dos augmented con freq/time masking
y otro par de image augmentation. Todos los espectrogramas de un mismo
paciente lo almazenaremos en una carpeta
'''

for idx, wav in enumerate(os.listdir(pos_path)):
    wav_path = os.path.join(pos_path, wav)
    # output_path = os.path.join(aug_path + 'pos/', wav.split('_')[0] + '_{}'.format(idx))
    output_path = os.path.join(aug_path, 'pos/')
    wav_to_spectrogram(wav_file_path=wav_path, spectrograms_path=output_path,augment_data=True, times_augmented=2)

for idx, spectr in enumerate(os.listdir(os.path.join(aug_path, 'pos/'))):

    spectr_path = os.path.join(os.path.join(aug_path, 'pos/'), spectr)
    image = tf.io.read_file(spectr_path)
    image = tf.image.decode_jpeg(image)

    times_augmented = 2

    for i in range(times_augmented):
        aug_image, aug_name = image_augmentation(image)
        output_path = spectr_path.strip('.jpg') + '_{}_augm_{}.jpg'.format(aug_name, i)
        plot_augmented_spectrogram(aug_image, output_path)



### FOR NEGATIVE COUGHS

for idx, wav in enumerate(os.listdir(neg_path)):
    wav_path = os.path.join(neg_path, wav)
    # output_path = os.path.join(aug_path + 'pos/', wav.split('_')[0] + '_{}'.format(idx))
    output_path = os.path.join(aug_path, 'neg/')
    wav_to_spectrogram(wav_file_path=wav_path, spectrograms_path=output_path,augment_data=True, times_augmented=2)

for idx, spectr in enumerate(os.listdir(os.path.join(aug_path, 'neg/'))):

    spectr_path = os.path.join(os.path.join(aug_path, 'neg/'), spectr)
    image = tf.io.read_file(spectr_path)
    image = tf.image.decode_jpeg(image)

    times_augmented = 2

    for i in range(times_augmented):
        aug_image, aug_name = image_augmentation(image)
        output_path = spectr_path.strip('.jpg') + '_{}_augm_{}.jpg'.format(aug_name, i)
        plot_augmented_spectrogram(aug_image, output_path)



'''
Let's make predictions with our model
'''

img_height, img_width, channels = 223, 223, 3
IMG_SIZE = (img_height, img_width)

newdata_path = aug_path
newdata_dataset = image_dataset_from_directory(newdata_path,image_size=IMG_SIZE)

model = tf.keras.models.load_model('C:/Users/Guillem/Desktop/New Coswara data/spectrograms/ResNet152V2_model_covid.h5')


loss, accuracy = model.evaluate(newdata_dataset)
print('New data accuracy :', accuracy)


### Create manually the dataset in order to parse all
### predictions of the same cough

filenames = []
for _pos in os.listdir(aug_path + 'pos/'):
    path = os.path.join(aug_path + 'pos/', _pos)
    filenames.append(path)

for _neg in os.listdir(aug_path + 'neg/'):
    path = os.path.join(aug_path + 'neg/', _neg)
    filenames.append(path)


pos_lab = np.ones(len(os.listdir(aug_path + 'pos/')))
neg_lab = np.zeros(len(os.listdir(aug_path + 'neg/')))
labels = np.concatenate([pos_lab, neg_lab]).tolist()



filenames = tf.constant(filenames)
labels = tf.constant(labels)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

def _parse_function(filename, label):
  image_string = tf.io.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  #image_resized = tf.image.resize_images(image_decoded, [233, 28])
  return filename, image_decoded, label

dataset = dataset.map(_parse_function)

'''
Since you trained your model on mini-batches, your input is a tensor of shape
[batch_size, image_width, image_height, number_of_channels].

When predicting, you have to respect this shape even if you have only one image.
Your input should be of shape: [1, image_width, image_height, number_of_channels]
'''

df = []
predictions = []
for filename, image, label in dataset:
    image_exp = np.expand_dims(image, axis = 0)
    pred = model.predict(image_exp)
    pred = tf.nn.sigmoid(pred).numpy()

    for _pred in pred:
        predictions.append(_pred)
        df.append([filename, _pred, label])

data = []
for filename, prediction, label in df:
    name = filename.numpy().decode("utf-8").strip('.jpg').split('/')[-1]
    data.append([name, prediction[0], label.numpy()])


corrected_filenames = []
for filename, pred, label in data:
    aux = filename.split('_')[0:2]
    name = aux[0] + '_' + aux[1]
    corrected_filenames.append(name)


df = pd.DataFrame(data, columns = ['filenames', 'prediction', 'label'])
df['filenames'] = corrected_filenames

df.to_excel('C:/Users/Guillem/Desktop/prueba.xlsx')


aggregate_df = df.groupby(['filenames'], sort = False, as_index=False).max()

pred = np.array(aggregate_df['prediction'])
lab = np.array(aggregate_df['label'])



Confusion_Matrix(y_true = lab, y_predicted = pred, pred_prob = True)











def image_augmentation(image):

    augmentation = ['rb','rc','rh','rcrop', 'rn', 'rs']
    _sample = random.sample(augmentation, k = 1)

    if _sample == 'rb':
        rb = tf.image.random_brightness(image, 0.2)
        return rb,_sample

    elif _sample == 'rc':
        rc = tf.image.random_contrast(image, 0.2, 0.5)
        return rc,_sample

    elif _sample == 'rh':
        rh = tf.image.random_hue(image, max_delta = 0.4)
        return rh,_sample

    elif _sample == 'rcrop':
        rcrop = tf.image.random_crop(image, size = [170, 170, 3])
        return rcrop,_sample

    elif _sample == 'rn':
        rn = tf.image.random_jpeg_quality(image, 10, 30)
        return rn,_sample

    else:
        rs = tf.image.random_saturation(image, 1, 2)
        return rs,_sample


def plot_augmented_spectrogram(aug_image, output_path, height = 224, width = 224):
    dpi = pylab.gcf().get_dpi()
    fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
    plt.imshow(aug_image.numpy().astype("uint8"))
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path, bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)


def wav_to_spectrogram(wav_file_path, spectrograms_path, augment_data = False, times_augmented = 1, height = 224, width = 224):
    dpi = pylab.gcf().get_dpi()
    name_file = os.path.basename(wav_file_path).split('.')[0] + '.jpg'
    output_path = os.path.join(spectrograms_path, name_file)

    with open(wav_file_path, 'rb') as f:
        waveform, sample_rate = librosa.load(f, sr = None)
        duration = len(waveform)/sample_rate

        if (duration < 1):
            f.close()
            os.remove(wav_file_path)
            return

        fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
        M = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        M_db = librosa.power_to_db(M, ref=np.max)
        # img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
        img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
        ax.set_ylim([0,5000])
        plt.axis('off')

        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(output_path, bbox_inches = 'tight',pad_inches = 0)
        plt.close(fig)

        if (augment_data == True):
            shape = M.shape
            M = np.reshape(M, (-1, shape[0], shape[1], 1))
            for i in range(times_augmented):

                name_file_augmented = os.path.basename(wav_file_path).split('.')[0] + '_augmented_{}.jpg'.format(i)
                output_path_augmented = os.path.join(spectrograms_path, name_file_augmented)

                mask_num = random.randint(1,5)
                freq_mask = random.randint(5,15)
                fm = frequency_masking(M, v = M.shape[0],frequency_masking_para=freq_mask, frequency_mask_num=mask_num)

                mask_num = random.randint(1,5)
                time_mask = random.randint(5,15)
                tm = time_masking(fm, tau=fm.shape[1], time_masking_para=time_mask, time_mask_num=mask_num)

                fig, ax = plt.subplots(figsize=(height/dpi, width/dpi))
                # img = librosa.display.specshow(librosa.power_to_db(tm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time', cmap='gray_r')
                img = librosa.display.specshow(librosa.power_to_db(tm[0, :, :, 0], ref=np.max), y_axis='mel', x_axis='time')
                ax.set_ylim([0,5000])
                plt.axis('off')

                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                plt.savefig(output_path_augmented, bbox_inches = 'tight',pad_inches = 0)
                plt.close(fig)

def frequency_masking(mel_spectrogram, v, frequency_masking_para=27, frequency_mask_num=2):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # Step 2 : Frequency masking
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=2):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, n-t0-t, v, 1)),
                          tf.zeros(shape=(1, t, v, 1)),
                          tf.ones(shape=(1, t0, v, 1)),
                          ), 1)
        mel_spectrogram = mel_spectrogram * mask

    return tf.cast(mel_spectrogram, dtype=tf.float32)


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
