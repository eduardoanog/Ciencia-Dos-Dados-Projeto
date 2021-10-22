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

import matplotlib.pyplot as plt

import sklearn.metrics as metrics
import numpy as np

import json

from tensorflow import keras
import time

import ast
import argparse

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

def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

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

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def getFileName(x):
    return os.path.split(x)[-1]

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Teste do modelo Transfer Learning')
    parser.add_argument('--BaseModel', type=str, help='BaseModel', required = True)
    parser.add_argument('--Dataset', type=str, help='Dataset', required = True)

    args = parser.parse_args()

    ##############################################################################
    #    Cria a pasta para armazenar os resultados e artefatos se não existir    #
    ##############################################################################
    if not(os.path.isdir(os.path.join('Resultados', args.Dataset, 'Transf',  args.BaseModel))):
        os.makedirs(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel))

    ##############################################################################
    #              Declaração dos parâmetros globais do código                   #
    ############################################################################## 
    PathTestData = os.path.join('Dataset', args.Dataset, 'test_set')

    ModelImgSize = (96, 96)

    Classes = ['Negative', 'Positive']
    ##############################################################################
    #      Faz a leitura dos caminhos das imagens e transforma em DataFrame      #
    ##############################################################################  
    TestFiles = []
    for path, subdirs, files in os.walk(PathTestData):
        for name in files:
            TestFiles.append(os.path.join(path, name))

    df_test = pd.DataFrame(TestFiles, columns = ['FilePath'])

    ##############################################################################
    #                Obtem qual é a classe que o caminho pertence                #
    ##############################################################################
    df_test['Class'] = df_test['FilePath'].apply(GetFileClass)

    ##############################################################################
    #               Fornece ao usuário a estatística de cada base                #
    ############################################################################## 
    print('Estatística do dataset de teste:')
    print(df_test['Class'].value_counts())

    ##############################################################################
    #        Carrega as imagens na memoria ram para acelerar o treinamento       #
    ##############################################################################  
    X_test, y_test, file_names_list = [], [], []

    df_test = df_test.reset_index(drop = True)

    print('Carregando as imagens de teste')
    for i in tqdm(range(len(df_test))):
        img = cv2.imread(df_test.loc[i, 'FilePath'])
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(im_rgb, ModelImgSize, interpolation = cv2.INTER_AREA)

        X_test.append(resized)
        y_test.append(df_test.loc[i, 'Class'])
        file_names_list.append(df_test.loc[i, 'FilePath'])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    ##############################################################################
    #                           Carrega o modelo                                 #
    ##############################################################################
    model = keras.models.load_model(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'transf.h5'))

    ##############################################################################
    #              Obtem o melhor threshold obtido no treinamento                #
    ##############################################################################
    with open(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'TrainFinalMetrics.json')) as f:
        TrainVariables = f.readlines()[0]
        TrainVariables = ast.literal_eval(TrainVariables)
    BestThreshold = TrainVariables['Threshold']

    ##############################################################################
    #  Faz o predict na base de teste e define o limiar que maximiza o Score F1  #
    ##############################################################################
    X_test_norm = X_test/255

    StartTime = time.time()
    y_pred_test = model.predict(X_test_norm)
    PredictTime = time.time() - StartTime
    y_test_real = y_test
    y_test_filenames = file_names_list.copy()

    df_predict_test = pd.DataFrame(y_test_filenames, columns = ['FilePaths'])
    df_predict_test['FileNames'] = df_predict_test['FilePaths'].apply(getFileName)
    df_predict_test['RealLabel'] = y_test_real
    df_predict_test['PredictScore'] = y_pred_test
    df_predict_test['PredictClass'] = to_labels(y_pred_test, BestThreshold)
    df_predict_test = df_predict_test.sort_values(by = 'FileNames')

    ####################################################################################
    #                        Obtem as métricas de validação                            #
    ####################################################################################
    FinalTrainMetrics = GetMetrics(y_pred_test, y_test_real, BestThreshold, args.BaseModel)
    FinalTrainMetrics['TempoPredict'] = PredictTime

    with open(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'TestFinalMetrics.json'), 'w') as fp:
        json.dump(FinalTrainMetrics, fp, default = np_encoder)

    print('Métricas de teste:')
    print(FinalTrainMetrics)

    ####################################################################
    #                       Plota a imagem de saída                    #
    ####################################################################
    DadosNegativeErrado = df_predict_test[(df_predict_test['RealLabel'] == 0) & (df_predict_test['RealLabel'] != df_predict_test['PredictClass'])].reset_index(drop = True)
    DadosPositiveErrado = df_predict_test[(df_predict_test['RealLabel'] == 1) & (df_predict_test['RealLabel'] != df_predict_test['PredictClass'])].reset_index(drop = True)
    
    Negative_len = min(3, len(DadosNegativeErrado))
    Positive_len = min(3, len(DadosPositiveErrado))

    f = plt.figure(figsize = (7, 3))
    for i in range(Negative_len):
        f.add_subplot(2, 3, i+1)
        img = cv2.cvtColor(cv2.resize(cv2.imread(DadosNegativeErrado['FilePaths'][i]), (64, 64)), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")
        plt.text(0, -15, 'True Class: ' + Classes[0] ,color = "b")
        plt.text(0, -3, 'Predicted Class: ' + Classes[DadosNegativeErrado['PredictClass'][i]] ,color = "r")

    for i in range(3, 3 + Positive_len):
        f.add_subplot(2, 3, i+1)
        img = cv2.cvtColor(cv2.resize(cv2.imread(DadosPositiveErrado['FilePaths'][i-3]), (64, 64)), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")
        plt.text(0, -15, 'True Class: ' + Classes[1] ,color = "b")
        plt.text(0, -3, 'Predicted Class: ' + Classes[DadosPositiveErrado['PredictClass'][i-3]] ,color = "r")
    plt.tight_layout()
    plt.savefig(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, "Resultado.png"))


    ####################################################################################
    #                         Salva os artefatos do predict                            #
    ####################################################################################
    df_predict_test.to_csv(os.path.join('Resultados', args.Dataset, 'Transf', args.BaseModel, 'TestArtifacts.csv'), sep = '\t', index = False)
