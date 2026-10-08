import os
import random
import shutil

semilla = 42
proporcion_training = 0.70
proporcion_validation = 0.15


def dividir_clase(ruta_clase, ruta_destino):
    clase = os.path.basename(ruta_clase)
    imagenes = sorted(os.listdir(ruta_clase))
    random.Random(semilla).shuffle(imagenes)

    num_training = int(len(imagenes) * proporcion_training)
    num_validation = int(len(imagenes) * proporcion_validation)

    particiones = {
        'training': imagenes[:num_training],
        'validation': imagenes[num_training:num_training + num_validation],
        'test': imagenes[num_training + num_validation:],
    }

    for conjunto, ficheros in particiones.items():
        carpeta = os.path.join(ruta_destino, conjunto, clase)
        os.makedirs(carpeta, exist_ok=True)
        for fichero in ficheros:
            shutil.copy2(os.path.join(ruta_clase, fichero), carpeta)

    print(f'{clase:10s} | training: {len(particiones["training"])} | '
          f'validation: {len(particiones["validation"])} | test: {len(particiones["test"])}')


if __name__ == '__main__':
    ruta_origen = 'dataset'
    ruta_destino = 'datos'

    for clase in sorted(os.listdir(ruta_origen)):
        dividir_clase(os.path.join(ruta_origen, clase), ruta_destino)
