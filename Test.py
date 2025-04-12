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


#def create_model():
    #model=RandomForestClassifier()

    #return model
class AUCCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(AUCCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # 验证数据
        x_val, y_val = self.validation_data
        # 预测
        y_pred = self.model.predict(x_val)
        # 注意：对于二分类问题，y_pred 应该是概率，而 y_val 是 0 或 1 的整数标签
        # 如果你的模型输出是 logits，你可能需要先通过 sigmoid 函数转换为概率
        if y_pred.shape[-1] > 1:  # 多分类情况
            y_pred = tf.nn.softmax(y_pred, axis=-1)[:, 1]  # 假设我们关注正类的概率
        else:
            y_pred = tf.sigmoid(y_pred)  # 二分类情况，转换为概率

        # 计算 AUC
        auc = roc_auc_score(y_val, y_pred.numpy().ravel())

        # 打印 AUC
        print(f'Epoch {epoch + 1}, Validation AUC: {auc:.4f}')

def create_model():

    model = Sequential()
    model.add(tf.keras.layers.Conv1D(32,3,3, input_shape=(768,1), activation='relu'))
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
    F1 = 2 * (PRE * REC) / (PRE + REC)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Acc, PRE, REC, F1, MCC
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data['feature']), np.array(data['label'])

def load_all_data(directory):

    X_train_neg, y_train_neg = load_data(os.path.join(directory, 'negative_train.json'))
    X_val_neg, y_val_neg = load_data(os.path.join(directory, 'negative_test.json'))
    X_train_pos, y_train_pos = load_data(os.path.join(directory, 'positive_train.json'))
    X_val_pos, y_val_pos = load_data(os.path.join(directory, 'positive_test.json'))

    X_train = np.concatenate((X_train_neg, X_train_pos), axis=0)
    y_train = np.concatenate((y_train_neg, y_train_pos), axis=0)
    X_test = np.concatenate((X_val_neg, X_val_pos), axis=0)
    y_test = np.concatenate((y_val_neg, y_val_pos), axis=0)

    print(X_train.shape)
    print(X_test.shape)

    return X_train, y_train,X_test, y_test
    #return X_train, y_train

if __name__ == '__main__':
    #  train train_label   X_test   y_test
    #  X_train  y_train    test     test_label
    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # 加载训练集,，读取Bert训练好的数据
    directory = 'm6A-SPP\data'
    X_train, y_train,X_test, y_test= load_all_data(directory)
    #X_train, y_train= load_all_data(directory)
    BATCH_SIZE = 16
    EPOCHS = 50
    model = create_model()
    validation_data = (X_test, y_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=validation_data,
                        callbacks=[AUCCallback(validation_data)])
    history.history
    with open('./one_log_history.txt', 'w') as f:
        f.write(str(history.history))

    train_loss = history.history["loss"]
    train_acc = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print('val loss:', loss)
    print('val accuracy:', accuracy)
    all_Acc = []
    all_PRE = []
    all_REC = []
    all_F1 = []
    all_MCC = []
    test_pred_all = []

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    test_pred = model.predict(X_test)
    # test_pred_all.append(test_pred[:, 1])

    # Sn, Sp, Acc, MCC, AUC
    Acc, PRE, REC, F1, MCC = show_performance(val_label, test_pred)
    AUC = roc_auc_score(val_label, test_pred)
    print('Acc = %f, PRE = %f, REC = %f, F1 = %f, MCC = %f' % (Acc, PRE, REC, F1, MCC))

    # Put each collapsed evaluation metric into a master list
    fpr, tpr, thresholds = roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, color='b', label=r'ROC (AUC=%0.4f)' % roc_auc, lw=2, alpha=.8)

    plt.plot(fpr, tpr, label='ROC cycle {} (AUC={:.4f})'.format(str(2), roc_auc))
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




