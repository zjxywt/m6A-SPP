import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, \
    classification_report

from keras.models import Model, load_model
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import utils

#with strategy.scope():
def create_model():
    model = Sequential()
    model.add(tf.keras.layers.Conv1D(32, 3, 3, input_shape=(768, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(1))
    model.add(tf.keras.layers.Conv1D(64, 3, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(1))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


    print(model.summary())
    return model

def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算PRE
    PRE = TP / (TP + FP + 1e-06)
    # 计算REC
    REC = TP / (TP + FN + 1e-06)
    # 计算F1
    F1 = 2*(PRE * REC) / (PRE + REC)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Acc, PRE, REC, F1, MCC

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data['feature']), np.array(data['label'])
    print(data['feature'])

def load_all_data(directory):

    X_train_pos, y_train_pos = load_data(os.path.join(directory, 'positive_train.json'))
    X_train_neg, y_train_neg = load_data(os.path.join(directory, 'negative_train.json'))


    print(X_train_neg.shape)
    print(X_train_pos.shape)
    X_train = np.concatenate((X_train_neg, X_train_pos), axis=0)
    y_train = np.concatenate((y_train_neg, y_train_pos), axis=0)

    return X_train, y_train


if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # 加载训练集,，读取Bert训练好的数据
    directory = 'm6A-SPP\data'
    X_train, y_train = load_all_data(directory)


    # Cross-validation
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)
    #k_fold =StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    #k_fold = KFold(n_splits=n, shuffle=False)
    sv_10_result = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    print(np.array(sv_10_result).shape)

    all_Acc = []
    all_PRE = []
    all_REC = []
    all_F1 = []
    all_MCC = []
    test_pred_all = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold_count, (train_index, val_index) in enumerate(k_fold.split(X_train)):
        print('*' * 30 + ' fold ' + str(fold_count) + ' ' + '*' * 30)
        trains, val = X_train[train_index], X_train[val_index]
        print(trains.shape)
        print(val.shape)
        trains_label, val_label = y_train[train_index], y_train[val_index]

        BATCH_SIZE = 16
        EPOCHS = 50
        model = create_model()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        history = model.fit(X_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(val, val_label))
        history.history
        with open('./one_log_history.txt', 'w') as f:
            f.write(str(history.history))

        train_loss = history.history["loss"]
        train_acc = history.history["accuracy"]
        val_loss = history.history["val_loss"]
        val_acc = history.history["val_accuracy"]

        loss, accuracy = model.evaluate(val, val_label, verbose=1)

        print('val loss:', loss)
        print('val accuracy:', accuracy)

        all_Acc = []
        all_PRE = []
        all_REC = []
        all_F1 = []
        all_MCC = []
        test_pred_all = []


        test_pred = model.predict(val)
            #test_pred_all.append(test_pred[:, 1])


        Acc, PRE, REC, F1, MCC = show_performance(val_label, test_pred)
        AUC = roc_auc_score(val_label, test_pred)
        print('Acc = %f, PRE = %f, REC = %f, F1 = %f, MCC = %f' % (Acc, PRE, REC, F1, MCC))

            # Put each collapsed evaluation metric into a master list
        fpr, tpr, thresholds = roc_curve(val_label, test_pred)
        roc_auc = auc(fpr, tpr)
        #plt.figure()
        #plt.plot(fpr, tpr, color='b', label=r'ROC (AUC=%0.4f)' % roc_auc, lw=2, alpha=.8)



        plt.plot(fpr, tpr, label='ROC cycle {} (AUC={:.4f})'.format(str(fold_count), roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC Curve of First Layer')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('ROC_Curve_of_test.jpg', dpi=1200, bbox_inches='tight')
        plt.legend(loc='lower right')



        all_Acc.append(Acc)
        all_PRE.append(PRE)
        all_REC.append(REC)
        all_F1.append(F1)
        all_MCC.append(MCC)
        fold_count += 1


        '''Mapping the ROC'''


    fold_avg_Acc = np.mean(all_Acc)
    fold_avg_PRE = np.mean(all_PRE)
    fold_avg_REC = np.mean(all_REC)
    fold_avg_F1 = np.mean(all_F1)
    fold_avg_MCC = np.mean(all_MCC)
    print("Acc: ", fold_avg_Acc)
    print("PRE: ", fold_avg_PRE)
    print("REC: ", fold_avg_REC)
    print("F1: ", fold_avg_F1)
    print("MCC: ", fold_avg_MCC)
    plt.show()




