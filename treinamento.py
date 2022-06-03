from urllib.request import urlretrieve
from deepface import DeepFace
from yoloface import face_analysis
from dlib import cnn_face_detection_model_v1
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from time import time
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt


def downloadDataset():
    api = KaggleApi()
    api.authenticate()

    # Criar o diretorio dataset_training
    if not os.path.exists('Banco Imagens'):
        os.makedirs('Banco Imagens')

    # Baixar o dataset de Yale Faces e salvar em dataset_training
    kaggle.api.dataset_download_files('asacxyz/ic-fatecitu', path='Banco Imagens', unzip=True)

    # Acessar o diretorio Banco Imagens e listar todas as imagens
    caminho = [os.path.join('Banco Imagens', f) for f in os.listdir('Banco Imagens')]

    for caminhoImagem in caminho:
        if ".jpeg" in caminhoImagem:
            nomeArquivo = os.path.splitext(caminhoImagem)[0]
            Image.open(caminhoImagem).convert('RGB').save(nomeArquivo + '.' + 'jpg')
            os.remove(caminhoImagem)


def prepararDataset(origem):

    diretorio = os.listdir(origem)

    cont = 1

    for caminhoImagem in diretorio:
        if ".jpg" in caminhoImagem:
            imagem = cv2.imread(caminhoImagem)
            img = cv2.resize(imagem, (600, 600))
            cv2.imwrite(caminhoImagem, img)
            # Renomear o arquivo
            os.rename(caminhoImagem, str(cont) + '-treinamento' + str(cont) + '.jpg')
            cont += 1


def criarArquivoComparativo(origem):
    os.chdir('..')
    origem = 'Banco Imagens'
    diretorio = os.listdir(origem)

    arquivo = open('dataset_sem_treino.txt', 'w')
    arquivo.write('Nome                         Positivo     Negativo')

    for imagem in diretorio:
        nome = imagem.split('-')[0]
        if int(nome) <= 400:
            if len(nome) == 1:
                arquivo.write('\n' + imagem + '              0            1')
            elif len(nome) == 2:
                arquivo.write('\n' + imagem + '            0            1')
            elif len(nome) == 3:
                arquivo.write('\n' + imagem + '          0            1')

        elif int(nome) > 400:
            if len(nome) == 1:
                arquivo.write('\n' + imagem + '              1            0')
            elif len(nome) == 2:
                arquivo.write('\n' + imagem + '            1            0')
            elif len(nome) == 3:
                arquivo.write('\n' + imagem + '          1            0')

    origem = os.chdir('Banco Imagens')

    return origem


def filtrarDataset(origem):
    y_train_verdadeiro = []
    diretorio = os.listdir(origem)
    for imagem in diretorio:
        nome = imagem.split('-')[0]
        if int(nome) <= 400:
            y_train_verdadeiro.append(0)
        elif int(nome) > 400:
            y_train_verdadeiro.append(1)
    return y_train_verdadeiro


def carregarImagemDeepface(origem):
    diretorio = os.listdir(origem)
    x_train = []
    for imagem in diretorio:
        img = cv2.imread(imagem)
        x_train.append(img)
    return x_train


def carregarImagem(origem):
    # sair do diretorio Banco Imagens
    origem = os.chdir('..')
    origem = 'Banco Imagens'
    diretorio = os.listdir(origem)
    arquivos = []
    for imagem in diretorio:
        arquivos.append('Banco Imagens/' + imagem)
    return arquivos


def detectarHaarCascade(x_train):
    y_train_predict = []
    tempos = []
    diretorio = x_train
    for imagem in diretorio:
        inicio = time()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        face = face_cascade.detectMultiScale(imagem, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))        
        if len(face) > 0:
            y_train_predict.append(1)
        else:
            y_train_predict.append(0)
        fim = time()
        tempos.append(fim - inicio)
    return y_train_predict, tempos


def detectarHog(x_train):
    y_train_predict = []
    tempos = []
    diretorio = x_train
    for imagem in diretorio:
        inicio = time()
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        face = hog.detectMultiScale(imagem)
        if len(face) > 0:
            y_train_predict.append(1)
        else:
            y_train_predict.append(0)
        fim = time()
        tempos.append(fim - inicio)
    return y_train_predict, tempos


def detectarDeepFace(x_train):
    y_train_predict = []
    tempos = []
    diretorio = x_train
    for imagem in diretorio:
        inicio = time()
        detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
        face = DeepFace.detectFace(imagem, detector_backend = detectors[4], enforce_detection = False)
        if face.all() != 0:
            y_train_predict.append(1)
        else:
            y_train_predict.append(0)
        fim = time()
        tempos.append(fim - inicio)
    return y_train_predict, tempos


def detectarCnn(x_train):
    y_train_predict = []
    tempos = []
    diretorio = x_train
    urlretrieve("https://github.com/justadudewhohacks/face-recognition.js-models/raw/master/models/mmod_human_face_detector.dat", "mmod_human_face_detector.dat")
    facial_detector = cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    for imagem in diretorio:
        inicio = time()
        face = facial_detector(imagem, 0)        
        if len(face) > 0:
            y_train_predict.append(1)                      
        else:
            y_train_predict.append(0)                
        fim = time()
        tempos.append(fim - inicio)
    return y_train_predict, tempos


def detectarYolo(x_train):
    y_train_predict = []
    tempos = []
    diretorio = x_train
    cont = 0
    print('Carregando os modelos de detecção .....')
    face=face_analysis()
    print('Fazendo detecção nas imagens ......')
    for imagem in diretorio:
        inicio = time()
        imagens, retangulos, grau_certeza = face.face_detection(image_path=imagem, model='full')
        if len(retangulos) > 0:
            y_train_predict.append(1)
        else:
            y_train_predict.append(0)        
        fim = time()
        tempos.append(fim - inicio)
    print('Fim da detecção.......')
    return y_train_predict, tempos


def resultados(y_train_verdadeiro, y_train_predict, tempos, nome):
    # Criar a matriz de confusão e mostrar ela na tela
    matriz = confusion_matrix(y_train_verdadeiro, y_train_predict)
    falsoNegativo, falsoPositivo, verdadeiroNegativo, verdadeiroPositivo = matriz.ravel()

    # Organizar os dados dentro da matriz confusão
    matrizConfusao = np.array([[falsoNegativo, falsoPositivo], [verdadeiroNegativo, verdadeiroPositivo]])

    # Visualização grafica da matriz de confusão
    sns.heatmap(matrizConfusao, annot=True, cmap='YlGnBu', fmt='g')
    plt.title('Matriz de Confusão do ' + nome)
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.show()

    # Gerar o grafico curva roc e mostrar na tela
    fpr, tpr, thresholds = metrics.roc_curve(y_train_verdadeiro, y_train_predict)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + nome)
    plt.show()

    maiorTempo = max(tempos)
    menorTempo = min(tempos)
    mediaTempo = sum(tempos) / len(tempos)
    tempoTotal = sum(tempos)

    # Visualização grafica dos tempos
    plt.plot(tempos)
    plt.title('Tempo de detecção ' + nome)
    plt.ylabel('Tempo')
    plt.xlabel('Imagens')
    plt.show()

    # Visualização grafico de barras de maior tempo, menor tempo, media de tempo e tempo total
    plt.bar(['Maior Tempo', 'Menor Tempo', 'Media de Tempo', 'Tempo Total'], [maiorTempo, menorTempo, mediaTempo, tempoTotal])
    plt.title('Tempo de detecção ' + nome)
    plt.ylabel('Tempo')
    plt.xlabel('Imagens')
    plt.show()

    # Mostrar a accuracia do algoritmo
    accuracy = accuracy_score(y_train_verdadeiro, y_train_predict)
    print('Accuracy ' + nome + ' : %.2f' % accuracy)

print('Fazendo o download do dataset .......')
downloadDataset()
origem = os.chdir('Banco Imagens')
print('Preparando o dataset .......')
prepararDataset(origem)
criarArquivoComparativo(origem)

decisao = 0

while decisao != 6:
    
    print('#'*10 + 'Detecção Facial' + '#'*10)
    print('Algoritmos sem Redes Neurais')
    print('[1] - Haar Cascade')
    print('[2] - Hog')
    print('Algoritmos com Redes Neurais')
    print('[3] - DeepFace')
    print('[4] - Cnn')
    print('[5] - Yolo')
    print('[6] - Sair')
    decisao = int(input('Digite a opção desejada: '))

    match decisao:
        case 1:
            nome = 'Haar Cascade'
            y_train_verdadeiro = filtrarDataset(origem)
            x_train = carregarImagemDeepface(origem)
            print('Haar Cascade')
            print('Iniciando detecção...')
            y_train_predict, tempos = detectarHaarCascade(x_train)
            print('Detecção finalizada')
            print('Resultados:')
            resultados(y_train_verdadeiro, y_train_predict, tempos, nome)

        case 2:
            nome = 'Hog'
            y_train_verdadeiro = filtrarDataset(origem)
            x_train = carregarImagemDeepface(origem)
            print('Hog')
            print('Iniciando detecção...')
            y_train_predict, tempos = detectarHog(x_train)
            print('Detecção finalizada')
            print('Resultados:')
            resultados(y_train_verdadeiro, y_train_predict, tempos, nome)

        case 3:
            nome = 'DeepFace'
            y_train_verdadeiro = filtrarDataset(origem)
            x_train = carregarImagemDeepface(origem)
            print('DeepFace')
            print('Iniciando detecção...')
            y_train_predict, tempos = detectarDeepFace(x_train)
            print('Detecção finalizada')
            print('Resultados:')
            resultados(y_train_verdadeiro, y_train_predict, tempos, nome)

        case 4:
            nome = 'Cnn'
            y_train_verdadeiro = filtrarDataset(origem)
            x_train = carregarImagemDeepface(origem)
            print('Cnn')
            print('Iniciando detecção...')
            y_train_predict, tempos = detectarCnn(x_train)
            print('Detecção finalizada')
            print('Resultados:')
            resultados(y_train_verdadeiro, y_train_predict, tempos, nome)

        case 5:
            nome = 'Yolo'
            y_train_verdadeiro = filtrarDataset(origem)
            x_train = carregarImagem(origem)
            print('Yolo')
            print('Iniciando detecção...')
            y_train_predict, tempos = detectarYolo(x_train)
            print('Detecção finalizada')
            print('Resultados:')
            resultados(y_train_verdadeiro, y_train_predict, tempos, nome)

        case 6:
            print('Saindo...')
            break
    
        
