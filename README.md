# Smartports CCTV - Clasificación de imágenes portuarias

Práctica 1 de la asignatura **AIDA**. Clasificación binaria de imágenes de cámaras CCTV de un puerto mediante redes convolucionales.

## Tareas

| Tarea | Imágenes | Clases |
|---|---|---|
| Ship/No-Ship | 294 | Barco presente / no presente |
| Docked/Undocked | 184 | Barco atracado / no atracado |

## Modelos

- **SimpleCNN** - CNN ligera entrenada desde cero (3 bloques conv)
- **MiniConvNeXt** - Arquitectura inspirada en ConvNeXt, desde cero
- **ResNet18** - Preentrenado en ImageNet, fine-tuning en dos fases

Cada modelo se evalúa con y sin *data augmentation* bajo **5-Fold Stratified CV**.

## Estructura

```
p1/
├── data/
│   ├── images/          # 294 imágenes JPEG
│   ├── ship.csv         # etiquetas Ship/No-Ship
│   └── docked.csv       # etiquetas Docked/Undocked
├── src/smartports/
│   ├── dataset.py       # SmartportsDataset
│   ├── models.py        # SimpleCNN, MiniConvNeXt, ResNet18
│   ├── transforms.py    # pipelines de augmentation
│   ├── train.py         # train_one_epoch, evaluate_loader, EarlyStopping
│   ├── evaluate.py      # métricas y figuras
│   └── experiment.py    # run_experiment, run_experiment_transfer
├── scripts/
│   ├── run_ship.py      # experimentos Ship/No-Ship
│   ├── run_docked.py    # experimentos Docked/Undocked
│   └── run_transfer.py  # transfer learning Ship → Docked
├── notebooks/
│   ├── eda.ipynb        # análisis exploratorio del dataset
│   └── analysis.ipynb   # análisis comparativo de resultados
├── outputs/
│   ├── checkpoints/     # pesos guardados por fold y tarea
│   ├── figures/         # gráficas generadas automáticamente
│   └── results/         # CSVs con métricas por fold
└── report/
    └── main.tex         # memoria en LaTeX
```

## Instalación y ejecución

```bash
# Instalar dependencias (requiere uv)
uv sync --group dev --group amd --prerelease=allow   # con GPU AMD en Windows
uv sync --group dev                                   # sin GPU

# Ejecutar experimentos en orden
uv run python scripts/run_ship.py
uv run python scripts/run_docked.py
uv run python scripts/run_transfer.py   # requiere haber ejecutado run_ship.py antes
```

## Hardware

| Componente | Especificación |
|---|---|
| CPU | AMD Ryzen 7 7800X3D (8 núcleos, Zen 4) |
| GPU | AMD Radeon RX 9070 XT (via torch-directml) |
| RAM | 48 GB |
| OS | Windows 11 Pro |
| PyTorch | 2.4.1 + torch-directml 0.2.5 |
| Python | 3.11 |

## Resultados principales

| Tarea | Mejor modelo | AUC-ROC |
|---|---|---|
| Ship/No-Ship | ResNet18 (sin aug) | 0.999 ± 0.001 |
| Docked/Undocked | ResNet18 (sin aug) | 0.861 ± 0.055 |

El transfer learning inter-tarea (Ship → Docked) no mejora los resultados base.
