from db_format import WavFileHelper
from network import NN
from globals import FULL_DATASET_PATH

import pandas as pd


def classifier():
    redo = input("Redo data preparation? (y/n)")
    helper = WavFileHelper(FULL_DATASET_PATH)
    if redo.lower() in ["y", "yes"]:
        print("Organizing data...")
        print("\t 1. Getting data props")
        props = helper.data_properties_df()
        props.to_csv('results/data_props.csv')
        print("\t 2. Getting features")
        features = helper.features_df()
        features.to_csv('results/features.csv')

    elif redo.lower() in ["n", "no"]:
        pass
    else:
        raise ValueError(f"Answer not recognized '{redo}'")

    print("\t 3. Getting training and testing data")
    yy, x_train, x_test, y_train, y_test = helper.get_train_data(features)
    print("Creating NN (y=")
    nn = NN(num_labels=yy.shape[1])
    print("\t Building NN")
    nn.build_nn()
    print("\t Adding data to NN")
    nn.add_data(x_train, x_test, y_train, y_test)
    print("\t Compiling NN")
    nn.compile_nn()
    print("\t Training NN")
    nn.train_nn()
    print("\t Evaluating NN")
    nn.evaluate_nn()



if __name__ == '__main__':
    classifier()
