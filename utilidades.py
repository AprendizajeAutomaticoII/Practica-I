import csv
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MEDIA = [0.6705, 0.6694, 0.6696]
DESVIACION = [0.1280, 0.1281, 0.1322]
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIACION_IMAGENET = [0.229, 0.224, 0.225]


def calcular_media_desviacion(ruta_training, tam_imagen, num_workers=4):
    transformaciones = transforms.Compose([
        transforms.Resize((tam_imagen, tam_imagen)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root=ruta_training, transform=transformaciones)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)

    suma = torch.zeros(3)
    suma_cuadrados = torch.zeros(3)
    num_pixeles = 0
    for imagenes, _ in dataloader:
        suma += imagenes.sum(dim=[0, 2, 3])
        suma_cuadrados += (imagenes ** 2).sum(dim=[0, 2, 3])
        num_pixeles += imagenes.numel() // 3

    media = suma / num_pixeles
    desviacion = torch.sqrt(suma_cuadrados / num_pixeles - media ** 2)
    return media, desviacion


def transformacion_base(tam_imagen, media=MEDIA, desviacion=DESVIACION):
    return transforms.Compose([
        transforms.Resize((tam_imagen, tam_imagen)),
        transforms.ToTensor(),
        transforms.Normalize(media, desviacion)
    ])


def crear_dataloaders(ruta, tam_imagen, tam_lote, transform_train=None, num_workers=4,
                      media=MEDIA, desviacion=DESVIACION):
    transformaciones = transformacion_base(tam_imagen, media, desviacion)
    if transform_train is None:
        transform_train = transformaciones

    dataset_entrenamiento = ImageFolder(root=os.path.join(ruta, 'training'), transform=transform_train)
    dataset_validacion = ImageFolder(root=os.path.join(ruta, 'validation'), transform=transformaciones)
    dataset_test = ImageFolder(root=os.path.join(ruta, 'test'), transform=transformaciones)

    persistente = num_workers > 0
    dataloader_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=tam_lote, shuffle=True,
                                          num_workers=num_workers, persistent_workers=persistente)
    dataloader_validacion = DataLoader(dataset_validacion, batch_size=tam_lote, shuffle=False,
                                       num_workers=num_workers, persistent_workers=persistente)
    dataloader_test = DataLoader(dataset_test, batch_size=tam_lote, shuffle=False, num_workers=num_workers)

    return dataloader_entrenamiento, dataloader_validacion, dataloader_test, dataset_entrenamiento.classes


def desnormalizar(imagen, media=MEDIA, desviacion=DESVIACION):
    media = torch.tensor(media).view(3, 1, 1)
    desviacion = torch.tensor(desviacion).view(3, 1, 1)
    imagen = imagen.cpu() * desviacion + media
    return imagen.clamp(0, 1).permute(1, 2, 0).numpy()


class ConvNet(nn.Module):
    def __init__(self, num_clases, tam_imagen=128, filtros=(16, 32, 64), neuronas_fc=(128,),
                 activacion=nn.ReLU, p_dropout=0.25):
        super().__init__()

        self.capas_convolucionales = nn.ModuleList()
        canales_entrada = 3
        for canales_salida in filtros:
            self.capas_convolucionales.append(nn.Sequential(
                nn.Conv2d(in_channels=canales_entrada, out_channels=canales_salida, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(canales_salida),
                activacion(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ))
            canales_entrada = canales_salida

        lado_final = tam_imagen // 2 ** len(filtros)
        self.tam_aplanado = filtros[-1] * lado_final * lado_final

        self.capas_fc = nn.ModuleList()
        neuronas_entrada = self.tam_aplanado
        for neuronas in neuronas_fc:
            self.capas_fc.append(nn.Sequential(
                nn.Linear(neuronas_entrada, neuronas),
                activacion(),
                nn.Dropout(p=p_dropout)
            ))
            neuronas_entrada = neuronas

        self.capa_salida = nn.Linear(neuronas_entrada, num_clases)

    def forward(self, x):
        for capa in self.capas_convolucionales:
            x = capa(x)

        x = x.view(-1, self.tam_aplanado)

        for capa in self.capas_fc:
            x = capa(x)

        return self.capa_salida(x)


def evaluar(modelo, dataloader, criterio=None):
    if criterio is None:
        criterio = nn.CrossEntropyLoss()

    modelo.eval()
    perdida_total = 0.0
    y_real = []
    y_pred = []

    with torch.no_grad():
        for imagenes, etiquetas in dataloader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            predicciones = modelo(imagenes)
            perdida_total += criterio(predicciones, etiquetas).item()
            y_pred.append(predicciones.argmax(dim=1).cpu().numpy())
            y_real.append(etiquetas.cpu().numpy())

    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    return y_real, y_pred, perdida_total / len(dataloader), accuracy_score(y_real, y_pred)


def entrenar(modelo, dataloader_entrenamiento, dataloader_validacion, lr, epocas_max, paciencia,
             ruta_modelo, criterio=None):
    if criterio is None:
        criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)

    historial = {'perdida_train': [], 'perdida_val': [], 'acc_train': [], 'acc_val': []}
    mejor_perdida_val = float('inf')
    contador_paciencia = 0
    comienzo = time()

    for epoca in range(epocas_max):
        modelo.train()
        perdida_total_epoca = 0.0
        y_real_train = []
        y_pred_train = []

        for imagenes, etiquetas in dataloader_entrenamiento:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)

            optimizador.zero_grad()
            predicciones = modelo(imagenes)
            perdida = criterio(predicciones, etiquetas)
            perdida.backward()
            optimizador.step()

            perdida_total_epoca += perdida.item()
            y_pred_train.append(predicciones.argmax(dim=1).detach().cpu().numpy())
            y_real_train.append(etiquetas.cpu().numpy())

        perdida_train = perdida_total_epoca / len(dataloader_entrenamiento)
        acc_train = accuracy_score(np.concatenate(y_real_train), np.concatenate(y_pred_train))
        _, _, perdida_val, acc_val = evaluar(modelo, dataloader_validacion, criterio)

        historial['perdida_train'].append(perdida_train)
        historial['perdida_val'].append(perdida_val)
        historial['acc_train'].append(acc_train)
        historial['acc_val'].append(acc_val)

        if perdida_val < mejor_perdida_val:
            mejor_perdida_val = perdida_val
            contador_paciencia = 0
            historial['mejor_epoca'] = epoca + 1
            torch.save(modelo.state_dict(), ruta_modelo)
            mejora = '*'
        else:
            contador_paciencia += 1
            mejora = ''

        print(f'Época [{epoca + 1:2d}/{epocas_max}] | '
              f'Train Loss: {perdida_train:.4f} | Train Acc: {acc_train:.4f} | '
              f'Val Loss: {perdida_val:.4f} | Val Acc: {acc_val:.4f} {mejora}')

        if contador_paciencia == paciencia:
            print(f'Parada anticipada: {paciencia} épocas sin mejorar la pérdida de validación')
            break

    historial['tiempo'] = time() - comienzo
    modelo.load_state_dict(torch.load(ruta_modelo))
    print(f'Mejor época: {historial["mejor_epoca"]} | Tiempo total: {historial["tiempo"]:.1f} segundos')
    return historial


def guardar_resultado(nombre_config, hiperparametros, historial, ruta_csv='resultados.csv', acc_test=None):
    mejor = historial['mejor_epoca'] - 1
    fila = {
        'configuracion': nombre_config,
        'hiperparametros': '; '.join(f'{clave}={valor}' for clave, valor in hiperparametros.items()),
        'epocas_entrenadas': len(historial['perdida_val']),
        'mejor_epoca': historial['mejor_epoca'],
        'perdida_train': round(historial['perdida_train'][mejor], 4),
        'perdida_val': round(historial['perdida_val'][mejor], 4),
        'acc_train': round(historial['acc_train'][mejor], 4),
        'acc_val': round(historial['acc_val'][mejor], 4),
        'acc_test': '' if acc_test is None else round(acc_test, 4),
        'tiempo_segundos': round(historial['tiempo'], 1),
    }

    existe = os.path.exists(ruta_csv)
    with open(ruta_csv, 'a', newline='', encoding='utf-8') as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=fila.keys())
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


def dibujar_curvas(historial, titulo, ruta_guardado=None):
    epocas_eje = range(1, len(historial['perdida_train']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epocas_eje, historial['perdida_train'], label='Entrenamiento')
    plt.plot(epocas_eje, historial['perdida_val'], label='Validación')
    plt.axvline(historial['mejor_epoca'], color='gray', linestyle='--', label='Mejor época')
    plt.title(f'Evolución de la pérdida - {titulo}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epocas_eje, historial['acc_train'], label='Entrenamiento')
    plt.plot(epocas_eje, historial['acc_val'], label='Validación')
    plt.axvline(historial['mejor_epoca'], color='gray', linestyle='--', label='Mejor época')
    plt.title(f'Evolución de la precisión - {titulo}')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if ruta_guardado is not None:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=120)
    plt.show()


def dibujar_matriz_confusion(y_real, y_pred, clases, titulo, ruta_guardado=None):
    matriz = confusion_matrix(y_real, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matriz, display_labels=clases).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Matriz de confusión - {titulo}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if ruta_guardado is not None:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=120)
    plt.show()


def ejecutar_configuracion(nombre_config, tam_imagen=128, tam_lote=32, lr=0.001, filtros=(16, 32, 64),
                           neuronas_fc=(128,), activacion=nn.ReLU, p_dropout=0.25, epocas_max=40,
                           paciencia=5, ruta_datos='datos', transform_train=None, criterio=None):
    torch.manual_seed(42)

    dataloader_entrenamiento, dataloader_validacion, _, clases = crear_dataloaders(
        ruta_datos, tam_imagen, tam_lote, transform_train)

    modelo = ConvNet(len(clases), tam_imagen, filtros, neuronas_fc, activacion, p_dropout).to(device)
    os.makedirs('modelos', exist_ok=True)
    historial = entrenar(modelo, dataloader_entrenamiento, dataloader_validacion, lr, epocas_max, paciencia,
                         os.path.join('modelos', f'{nombre_config}.pth'), criterio)

    hiperparametros = {
        'tam_imagen': tam_imagen,
        'tam_lote': tam_lote,
        'lr': lr,
        'filtros': filtros,
        'neuronas_fc': neuronas_fc,
        'activacion': activacion.__name__,
        'p_dropout': p_dropout,
        'aumentado': transform_train is not None,
        'pesos_clase': criterio is not None,
    }
    guardar_resultado(nombre_config, hiperparametros, historial)
    dibujar_curvas(historial, nombre_config, os.path.join('graficas', f'curvas_{nombre_config}.png'))
    return modelo, historial
