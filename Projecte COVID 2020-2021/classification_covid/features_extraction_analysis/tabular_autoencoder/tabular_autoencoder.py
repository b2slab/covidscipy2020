import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('C:/Users/Guillem/Desktop/Anomaly detection (autoencoders)/python scripts/features_metadata.xlsx', index_col=0)
df.shape

df = df.drop('filename', axis=1)

# splitting by class
pos = df[df.label == 1]
neg = df[df.label == 0]

# undersample neg
neg_undersampled = neg.sample(int(len(pos)))

# concatenate with fraud transactions into a single dataframe
visualisation_initial = pd.concat([pos, neg_undersampled])
column_names = list(visualisation_initial.drop(['label'], axis=1).columns)

# isolate features from labels
features, labels = visualisation_initial.drop('label', axis=1).values, \
                   visualisation_initial.label.values


from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def tsne_scatter(features, labels, dimensions=2):
    if dimensions not in (2, 3):
        raise ValueError('tsne_scatter can only plot in 2d or 3d (What are you? An alien that can visualise >3d?). Make sure the "dimensions" argument is in (2, 3)')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions).fit_transform(features)

    # initialising the plot
    fig, ax = plt.subplots(figsize=(8,8))

    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels==1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Pos'
    )

    ax.scatter(
        *zip(*features_embedded[np.where(labels==0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Neg'
    )

    # storing it to be displayed later
    plt.legend(loc='best')
    #plt.savefig(save_as)
    plt.show

tsne_scatter(features, labels, dimensions=2)


TRAINING_SAMPLE = int(len(neg)*0.935)

# shuffle our training set
neg = neg.sample(frac=1).reset_index(drop=True)

# training set: exlusively non-fraud transactions
X_train = neg.iloc[:TRAINING_SAMPLE].drop('label', axis=1)

# testing  set: the remaining non-fraud + all the fraud
X_test = neg.iloc[TRAINING_SAMPLE:].append(pos).sample(frac=1)

X_train.shape
X_test.shape
X_test.label.value_counts()


from sklearn.model_selection import train_test_split
VALIDATE_SIZE = 0.1
RANDOM_SEED = 1234
# train // validate - no labels since they're all clean anyway
X_train, X_validate = train_test_split(X_train,
                                       test_size=VALIDATE_SIZE,
                                       random_state=RANDOM_SEED)

# manually splitting the labels from the test df
X_test, y_test = X_test.drop('label', axis=1).values, X_test.label.values


from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline

# configure our pipeline
pipeline = Pipeline([('normalizer', Normalizer()),
                     ('scaler', MinMaxScaler())])

# get normalization parameters by fitting to the training data
pipeline.fit(X_train);

# transform the training and validation data with these parameters
X_train_transformed = pipeline.transform(X_train)
X_validate_transformed = pipeline.transform(X_validate)


g = sns.PairGrid(X_train.iloc[:,:3].sample(600, random_state=RANDOM_SEED))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Before:')
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot);

g = sns.PairGrid(pd.DataFrame(X_train_transformed, columns=column_names).iloc[:,:3].sample(600, random_state=RANDOM_SEED))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('After:')
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot);





# data dimensions // hyperparameters
input_dim = X_train_transformed.shape[1]
BATCH_SIZE = 32
EPOCHS = 500

# https://keras.io/layers/core/
autoencoder = tf.keras.models.Sequential([

    # deconstruct / encode
    tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(2, activation='elu'),

    # reconstruction / decode
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(input_dim, activation='elu')

])


autoencoder.compile(optimizer="adam",
                    loss="mse",
                    metrics=["acc"])

# print an overview of our model
autoencoder.summary()

# define our early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

save_model = tf.keras.callbacks.ModelCheckpoint(
    filepath='C:/Users/Guillem/Desktop/Anomaly detection (autoencoders)/audios/tabular_autoencoder.hdf5',
    save_best_only=True,
    monitor='val_loss',
    verbose=0,
    mode='min'
)

# callbacks argument only takes a list
cb = [early_stop, save_model]


history = autoencoder.fit(
    X_train_transformed, X_train_transformed,
    shuffle=True,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    validation_data=(X_validate_transformed, X_validate_transformed)
);


# transform the test set with the pipeline fitted to the training set
X_test_transformed = pipeline.transform(X_test)

# pass the transformed test set through the autoencoder to get the reconstructed result
reconstructions = autoencoder.predict(X_test_transformed)



# calculating the mean squared error reconstruction loss per row in the numpy array
mse = np.mean(np.power(X_test_transformed - reconstructions, 2), axis=1)
mse.shape

negative = mse[y_test==0]
positive = mse[y_test==1]

fig, ax = plt.subplots(figsize=(6,6))

ax.hist(negative, bins=50, density=True, label="neg", alpha=.6, color="green")
ax.hist(positive, bins=50, density=True, label="pos", alpha=.6, color="red")

plt.title("(Normalized) Distribution of the Reconstruction Loss")
plt.legend()
plt.show()



THRESHOLD = 3

def mad_score(points):
    """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)

    return 0.6745 * ad / mad

z_scores = mad_score(mse)
outliers = z_scores > THRESHOLD

print(f"Detected {np.sum(outliers):,} outliers in a total of {np.size(z_scores):,} transactions [{np.sum(outliers)/np.size(z_scores):.2%}].")



negatives = z_scores[y_test==0]
positives = z_scores[y_test==1]

fig, ax = plt.subplots(figsize=(6,6))

ax.hist(negatives, bins=50, density=True, label="neg", alpha=.6, color="green")
ax.hist(positives, bins=50, density=True, label="pos", alpha=.6, color="red")

plt.title("Distribution of the modified z-scores")
plt.legend()
plt.show()
