# Practica-I – Clasificación de tallas de diamante con redes convolucionales

Aprendizaje Automático II 2026-27 · Daniel y Amado

Clasificador CNN en PyTorch de 7 tallas de diamante (cushion, emerald, heart, oval, pear, princess, round), con 3391 imágenes por clase. La clase marquise (465 imágenes) se usa aparte para estudiar el efecto de tener menos imágenes.

## Puesta en marcha

1. Clonar el repositorio y descomprimir el zip del dataset dentro, de forma que quede `Practica-I/dataset/<clase>/` y `Practica-I/marquise/`.
2. Instalar las librerías en el entorno virtual:
   ```
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   python -m pip install matplotlib scikit-learn pillow ipykernel
   ```
3. Generar la partición 70/15/15 (siempre la misma, con semilla 42):
   ```
   python particion_dataset.py
   ```
   Crea `datos/training`, `datos/validation` y `datos/test` con 2373 / 508 / 510 imágenes por clase.

`dataset/`, `marquise/`, `datos/`, `datos_marquise/` y `modelos/*.pth` no se suben al repositorio.

## Archivos

| Archivo | Responsable | Contenido |
| --- | --- | --- |
| `particion_dataset.py` | Daniel | División en training, validation y test |
| `utilidades.py` | Daniel | Carga de datos, `ConvNet`, entrenamiento con parada anticipada, evaluación, gráficas y registro de resultados |
| `1-exploracion-datos.ipynb` | Daniel | Tamaños, distribución por clase, ejemplos y normalización |
| `2-busqueda-manual.ipynb` | Daniel (C1–C4) y Amado (C5–C8) | Búsqueda manual de hiperparámetros |
| `3-modelo-final.ipynb` | Amado | Mejor configuración evaluada en test |
| `4-data-augmentation.ipynb` | Amado | Aumentado de datos |
| `5-transfer-learning.ipynb` | Daniel | ResNet18 y MobileNetV2 preentrenadas |
| `6-tuning-automatico.ipynb` | Amado | Búsqueda automática de hiperparámetros |
| `7-marquise.ipynb` | Daniel | Experimento con la clase marquise |
| `resultados.csv` | Automático | Una fila por cada entrenamiento, con el dispositivo en el que se ejecutó (GPU o CPU). `acc_test` solo está en los modelos evaluados en test |
| `graficas/` | Automático | Curvas y matrices de confusión |

## Uso de `utilidades.py`

Una configuración completa (carga de datos, red, entrenamiento con parada anticipada, fila en `resultados.csv` y curvas en `graficas/`):

```python
import torch.nn as nn
from utilidades import ejecutar_configuracion

modelo, historial = ejecutar_configuracion('C5', filtros=(32, 64, 128, 256))
modelo, historial = ejecutar_configuracion('C7', activacion=nn.LeakyReLU)
```

Parámetros de `ejecutar_configuracion` y su valor por defecto: `tam_imagen=128`, `tam_lote=32`, `lr=0.001`, `filtros=(16, 32, 64)`, `neuronas_fc=(128,)`, `activacion=nn.ReLU`, `p_dropout=0.25`, `epocas_max=40`, `paciencia=5`, `ruta_datos='datos'`, `transform_train=None` (para el aumentado de datos) y `criterio=None` (para una pérdida distinta).

Evaluación en test de un modelo ya entrenado:

```python
from utilidades import crear_dataloaders, evaluar, dibujar_matriz_confusion

_, _, dataloader_test, clases = crear_dataloaders('datos', 128, 32)
y_real, y_pred, perdida_test, acc_test = evaluar(modelo, dataloader_test)
dibujar_matriz_confusion(y_real, y_pred, clases, 'Modelo final', 'graficas/matriz_final.png')
```

Aumentado de datos solo en training: construir el `transforms.Compose` con los aumentos y terminar con `transforms.ToTensor()` y `transforms.Normalize(MEDIA, DESVIACION)`, y pasarlo como `transform_train`.

Si el ordenador tiene pocos núcleos o poca memoria, bajar `num_workers` en `crear_dataloaders` (por defecto 4). Cada proceso de carga ocupa unos 300 MB en Windows.
