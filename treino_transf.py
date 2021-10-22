##############################################################################
#                      Importa as bibliotecas                                #
##############################################################################
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import sklearn.metrics as metrics
import numpy as np

import json

import time

import argparse
from sklearn.utils import class_weight

##############################################################################
#                            Declara as funções                              #
##############################################################################
def GetFileClass(x):
    # Retorna se o caminho da imagem é da classe positive ou negative
    # Parameters:
    #    x(str): String com o caminho da imagem.
    # Returns:
    #    (str): String com a classe correta do caminho.   
    if 'Positive' in x:
        return 1
    else:
        return 0

def PlotTrainHistory(history, filename):
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['val_accuracy'], 'r', label = 'Acurácia de Validação')
    plt.plot(history.history['val_loss'], 'r--', label = 'Loss de Validação - binary_crossentropy')
    plt.plot(history.history['accuracy'], 'b', label = 'Acurácia do Treino')
    plt.plot(history.history['loss'], 'b--', label = 'Loss do Treino - binary_crossentropy')
    plt.title('Loss e Acurácia - Treinamento e validação')
    plt.ylabel('Loss e Acurácia')
    plt.xlabel('Épocas do treinamento')
    plt.ylim(0)
    plt.legend(loc = 'best')
    plt.savefig(filename)

def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

def GetBestTreshold(y_real, y_pred):
    thresholds = np.arange(0, 1, 0.001)
    scores = [metrics.f1_score(y_real, to_labels(y_pred, t)) for t in thresholds]
    ix = np.argmax(scores)
    return thresholds[ix]

def GetMyF1(y_test, y_pred, t):
    return metrics.f1_score(list(y_test), to_labels(list(y_pred), t), zero_division = 0)

def GetMyRE(y_test, y_pred, t):
    return metrics.recall_score(list(y_test), to_labels(list(y_pred), t), zero_division = 0)

def GetMyPR(y_test, y_pred, t):
    return metrics.precision_score(list(y_test), to_labels(list(y_pred), t), zero_division = 0)

def GetMyACC(y_test, y_pred, t):
    return metrics.accuracy_score(list(y_test), to_labels(list(y_pred), t))

def GetMetrics(y_pred, y_test, threshold, ModeloBase):
    fpr, tpr, thresholds = metrics.roc_curve(list(y_test), list(y_pred))
    roc_auc = metrics.auc(fpr, tpr)
    ypred_thresholded = to_labels(y_pred,  threshold)
    tn, fp, fn, tp = metrics.confusion_matrix(list(y_test), ypred_thresholded).ravel()
    acc = ((tp + tn)/(tn + fp + fn + tp))
    pr = tp/(tp + fp)
    re = tp/(tp + fn)
    f1 = 2*((pr*re)/(pr+re))
    return {'Threshold': threshold, 'ModeloBase': ModeloBase, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'ACC': acc, 'PR': pr, 'RE': re, 'F1': f1, 'AUC': roc_auc, 'Erro': 1 - acc}

def PlotCurves(y_test, y_pred, filenameA, filenameB, threshold):
    fpr, tpr, thresholds = metrics.roc_curve(list(y_test), list(y_pred))
    roc_auc = metrics.auc(fpr, tpr)
    thresholds = list(np.linspace(0, 0.99, 100))
    scoresF1 = [GetMyF1(y_test, y_pred, t) for t in tqdm(thresholds)]
    scoresRE = [GetMyRE(y_test, y_pred, t) for t in tqdm(thresholds)]
    scoresPR = [GetMyPR(y_test, y_pred, t) for t in tqdm(thresholds)]
    scoresACC = [GetMyACC(y_test, y_pred, t) for t in tqdm(thresholds)]

    plt.figure(figsize = (4,3))
    plt.title('Curva ROC/AUC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc, color = 'b')
    plt.legend(loc = 'best')
    plt.xlabel('Taxa de Verdadeiros Positivos')
    plt.ylabel('Taxa de Falso Positivos')
    plt.tight_layout()
    plt.savefig(filenameA)

    plt.figure(figsize = (4,3))
    plt.title('Curvas RE/PR/F1/ACC')
    plt.plot(thresholds, scoresPR, label = 'Precisão', color = 'b')
    plt.plot(thresholds, scoresRE, label = 'Revocação', color = 'r')
    plt.plot(thresholds, scoresF1, label = 'F1', color = 'g')
    plt.plot(thresholds, scoresACC, label = 'Acurácia', color = 'y')
    plt.axvline(x = threshold, label = 'Melhor Threshold', color = 'k')
    plt.legend(loc = 'best')
    plt.xlabel('Threshold')
    plt.ylabel('Métricas Pr/Re/F1')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(filenameB)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def GetBaseModel(ModelName, InputSize, trainBase):
    if ModelName == 'DenseNet121':
        base_model = tf.keras.applications.DenseNet121(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')
    
    elif ModelName == 'DenseNet169':
        base_model = tf.keras.applications.DenseNet169(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'DenseNet201':
        base_model = tf.keras.applications.DenseNet201(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')
                                                

    elif ModelName == 'InceptionV3':
        base_model = tf.keras.applications.InceptionV3(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'ResNet101':
        base_model = tf.keras.applications.ResNet101(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'ResNet101V2':
        base_model = tf.keras.applications.ResNet101V2(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'ResNet152':
        base_model = tf.keras.applications.ResNet152(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'ResNet152V2':
        base_model = tf.keras.applications.ResNet152V2(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'ResNet50V2':
        base_model = tf.keras.applications.ResNet50V2(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'VGG16':
        base_model = tf.keras.applications.VGG16(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'VGG19':
        base_model = tf.keras.applications.VGG19(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    elif ModelName == 'Xception':
        base_model = tf.keras.applications.Xception(input_shape = InputSize,
                                                include_top = False,
                                                weights = 'imagenet')

    base_model.trainable = trainBase
    return base_model


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Treino do modelo Transfer Learning')
    parser.add_argument('--BaseModel', type=str, help='BaseModel', required = True)
    parser.add_argument('--Dataset', type=str, help='Dataset', required = True)

    args = parser.parse_args()

    ##############################################################################
    #            Variável de análise se vai ser treinado o modelo base           #
    ##############################################################################
    TrainBase = False

    ##############################################################################
    #    Cria a pasta para armazenar os resultados e artefatos se não existir    #
    ##############################################################################
    if not(os.path.isdir(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel))):
        os.makedirs(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel))

    ##############################################################################
    #              Declaração dos parâmetros globais do código                   #
    ##############################################################################       
    # Caminhos para as bases de teste e treino
    PathTrainData = os.path.join('Dataset', args.Dataset, 'training_set')

    # Tamanho da imagem para o modelo
    InputSize = (96, 96, 3)
    ModelImgSize = (96, 96)

    ##############################################################################
    #      Faz a leitura dos caminhos das imagens e transforma em DataFrame      #
    ##############################################################################  
    TrainFiles = []
    for path, subdirs, files in os.walk(PathTrainData):
        for name in files:
            TrainFiles.append(os.path.join(path, name))

    df_train = pd.DataFrame(TrainFiles, columns = ['FilePath'])

    ##############################################################################
    #                Obtem qual é a classe que o caminho pertence                #
    ##############################################################################  
    df_train['Class'] = df_train['FilePath'].apply(GetFileClass)

    ##############################################################################
    #          Cria um subset de validação apartir da base de treinamento        #
    ##############################################################################
    df_train, df_val = train_test_split(df_train, test_size = 0.2, random_state = 42)

    ##############################################################################
    #               Fornece ao usuário a estatística de cada base                #
    ##############################################################################  
    print('Estatística do dataset de treino:')
    print(df_train['Class'].value_counts())

    print('Estatística do dataset de validação:')
    print(df_val['Class'].value_counts())

    ##############################################################################
    #        Carrega as imagens na memoria ram para acelerar o treinamento       #
    ##############################################################################  
    X_train, X_val, y_train, y_val = [], [], [], []

    df_train = df_train.reset_index(drop = True)
    df_val = df_val.reset_index(drop = True)

    print('Carregando as imagens de treinamento')
    for i in tqdm(range(len(df_train))):
        img = cv2.imread(df_train.loc[i, 'FilePath'])
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(im_rgb, ModelImgSize, interpolation = cv2.INTER_AREA)

        X_train.append(resized)
        y_train.append(df_train.loc[i, 'Class'])

    print('Carregando as imagens de validação')
    for i in tqdm(range(len(df_val))):
        img = cv2.imread(df_val.loc[i, 'FilePath'])
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(im_rgb, ModelImgSize, interpolation = cv2.INTER_AREA)

        X_val.append(resized)
        y_val.append(df_val.loc[i, 'Class'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_val:', X_val.shape)
    print('y_val:', y_val.shape)

    ##############################################################################
    #          Define as classes de data augmentation e normalização             #
    ##############################################################################  
    TrainDatagen = ImageDataGenerator(rotation_range = 60,
                                      width_shift_range = 0.2,
                                      height_shift_range = 0.2,
                                      rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True,
                                      fill_mode = 'nearest')

    TestValDatagen = ImageDataGenerator(rescale=1./255)

    TrainDatagen.fit(X_train)
    TestValDatagen.fit(X_val)

    ##############################################################################
    #                             Define o modelo                                #
    ##############################################################################
    base_model = GetBaseModel(args.BaseModel, InputSize, TrainBase)

    inputs = keras.Input(shape = InputSize)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation = 'relu')(x)
    x = keras.layers.Dense(1024, activation = 'relu')(x)
    x = keras.layers.Dropout(0.8)(x)
    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())

    ##############################################################################
    #                   Define o critério de parada do modelo                    #
    ##############################################################################
    callbacks_list = [keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                    patience = 5,
                                                    restore_best_weights = True)]

    ##############################################################################
    #         Define os pesos de acordo com o desbalanço das classes             #
    ##############################################################################
    weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {i : weight[i] for i in range(2)}

    ##############################################################################
    #                             Treina o modelo                                #
    ##############################################################################
    StartTime = time.time()
    history = model.fit(TrainDatagen.flow(X_train, y_train, batch_size = 16),
                        epochs = 999,
                        validation_data = TestValDatagen.flow(X_val, y_val, batch_size = 16),
                        workers = -1,
                        callbacks = callbacks_list,
                        verbose = 1,
                        class_weight = class_weights)
    TrainingTime = time.time() - StartTime

    #############################################################################
    #                              Salva o modelo                               #
    #############################################################################
    model.save(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'transf.h5'))

    #############################################################################
    #                Plota o histórico de treinamento e validação               #
    #############################################################################
    PlotTrainHistory(history, os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'TrainHistory.png'))

    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'TrainHistory.csv'), sep = '\t')
    
    ##################################################################################
    #                            Libera memoria ram                                  #
    ##################################################################################
    del X_train, y_train

    ##################################################################################
    #  Faz o predict na base de validação e define o limiar que maximiza o Score F1  #
    ##################################################################################
    X_val_norm = X_val/255
    y_pred_val = model.predict(X_val_norm)
    y_val_real = y_val

    BestThreshold = GetBestTreshold(y_val_real, y_pred_val)

    ####################################################################################
    #          Imprime a curva ROC/AUC e a curva dos Scores F1/RE/PR/ACC               #
    ####################################################################################
    PlotCurves(y_val_real, y_pred_val, os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'ROCAUC.png'), os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'MetricCurves.png'), BestThreshold)

    ####################################################################################
    #                        Obtem as métricas de validação                            #
    ####################################################################################
    FinalTrainMetrics = GetMetrics(y_pred_val, y_val_real, BestThreshold, args.BaseModel)
    FinalTrainMetrics['TempoTreinamento'] = TrainingTime

    with open(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'TrainFinalMetrics.json'), 'w') as fp:
        json.dump(FinalTrainMetrics, fp, default = np_encoder)

    print('Métricas de validação:')
    print(FinalTrainMetrics)

    ####################################################################################
    #                         Salva os artefatos do predict                            #
    ####################################################################################
    df_predict_artifact = pd.DataFrame(y_pred_val, columns = ['pred'])
    df_predict_artifact['real'] = y_val_real
    df_predict_artifact.to_csv(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'ValidationPredict.csv'), sep = '\t', index = False)
