import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

MEDIA = [0.6705, 0.6694, 0.6696]
DESVIACION = [0.1280, 0.1281, 0.1322]


def calcular_media_desviacion(ruta_training, tam_imagen, num_workers=6):
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


def transformacion_base(tam_imagen):
    return transforms.Compose([
        transforms.Resize((tam_imagen, tam_imagen)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIACION)
    ])


def crear_dataloaders(ruta, tam_imagen, tam_lote, transform_train=None, num_workers=6):
    transformaciones = transformacion_base(tam_imagen)
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
    dataloader_test = DataLoader(dataset_test, batch_size=tam_lote, shuffle=False,
                                 num_workers=num_workers, persistent_workers=persistente)

    return dataloader_entrenamiento, dataloader_validacion, dataloader_test, dataset_entrenamiento.classes


def desnormalizar(imagen):
    media = torch.tensor(MEDIA).view(3, 1, 1)
    desviacion = torch.tensor(DESVIACION).view(3, 1, 1)
    imagen = imagen * desviacion + media
    return imagen.clamp(0, 1).permute(1, 2, 0).numpy()
