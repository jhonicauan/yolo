# Detecção de Etiquetas + Análise de Defeitos

Pipeline completa pra detectar etiquetas em imagens/vídeo, planificá-las e classificar como boas ou defeituosas.

**Estágios:**

1. **YOLO-OBB** detecta a etiqueta com bounding box rotacionada.
2. **SAM** (Segment Anything) refina a borda em pixel-perfect.
3. **Perspective warp** planifica a etiqueta como se tivesse sido escaneada frontalmente.
4. **EfficientAd (anomalib)** classifica o crop como normal ou anômalo, com score de 0 a 100 %.

Modelos treinados prontos em `models/`. Não precisa treinar pra usar.

---

## Estrutura

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── models/                 # Modelos treinados e pré-treinados
│   ├── yolo_obb.pt         #   YOLO-OBB treinado para etiquetas
│   ├── anomalib.ckpt       #   EfficientAd treinado para defeitos
│   └── mobile_sam.pt       #   MobileSAM (pretrained)
├── configs/                # Configs de tracker
│   ├── bytetrack.yaml
│   └── botsort.yaml
├── samples/                # Imagens de exemplo
│   ├── correto.jpeg
│   └── errado.jpeg
└── src/
    ├── pipeline.py         # Utilitários (SAM, warp, caminhos)
    ├── train_yolo.py       # Treinar YOLO-OBB
    ├── train_anomalib.py   # Treinar EfficientAd
    ├── predict_image.py    # YOLO em imagem/pasta/vídeo
    ├── predict_camera.py   # YOLO ao vivo (webcam)
    ├── track_camera.py     # YOLO + tracker + contagem
    ├── predict_anomaly.py  # Pipeline completa (imagem -> score de defeito)
    └── data_prep/
        ├── import_labelstudio.py    # Importa export do Label Studio
        ├── extract_annotated.py     # Extrai imagens anotadas do zip
        ├── extract_backgrounds.py   # Extrai backgrounds do zip
        └── prepare_anomaly_crops.py # Gera crops planificados pro anomalib
```

---

## Instalação

### Requisitos

- Python 3.10 ou 3.11
- Windows / Linux / macOS

### Passo 1 — instalar dependências

**Com GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Sem GPU (CPU-only):**
```bash
pip install -r requirements.txt
```

### Passo 2 — verificar

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Como usar (modelos já treinados)

Todos os scripts detectam CPU/GPU automaticamente. Use `--device cpu` ou `--device cuda` pra forçar.

### Analisar uma imagem completa (pipeline inteira)

```bash
python src/predict_anomaly.py samples/errado.jpeg
```

Saída no terminal:
```
Device      : cuda
Anomalib    : models/anomalib.ckpt
YOLO OBB    : models/yolo_obb.pt
Detectou 2 etiqueta(s).

  crop 0:  87.42%  [ANOMALO]
  crop 1:  12.31%  [normal]

Heatmaps salvos em: runs/anomaly/
```

Cada crop gera um PNG lado-a-lado (original + heatmap) em `runs/anomaly/`.

### Analisar um crop já planificado

```bash
python src/predict_anomaly.py --crop crops_dataset/test/defeito/errado__0.png
```

### Só YOLO em imagem/pasta/vídeo

```bash
python src/predict_image.py samples/errado.jpeg
python src/predict_image.py minha_pasta/
python src/predict_image.py video.mp4 --conf 0.6
```

### YOLO ao vivo (webcam)

```bash
python src/predict_camera.py
python src/predict_camera.py --sam --sam-mode quad
```

### Tracking + contagem

```bash
python src/track_camera.py
python src/track_camera.py --source video.mp4 --save saida.mp4
python src/track_camera.py --tracker botsort
```

### Interpretação do score

- **0–30 %** → provavelmente normal
- **30–70 %** → incerto, revisar manualmente
- **70–100 %** → provavelmente com defeito

Padrão: score > 0.5 = anômalo. Aumente com `--threshold 0.7` pra reduzir falsos positivos.

---

## Retreinar (opcional)

### YOLO-OBB

1. Anotar imagens no **Label Studio** com **Rotated Rectangle**.
2. Exportar como **YOLOv8 OBB with Images** → `export.zip`.
3. Importar:
   ```bash
   python src/data_prep/import_labelstudio.py export.zip --dataset etiquetas --clean
   ```
4. Treinar:
   ```bash
   python src/train_yolo.py etiquetas --epochs 100
   ```
5. Copiar melhor peso pra pasta de modelos:
   ```bash
   cp runs/obb/obb-etiquetas/weights/best.pt models/yolo_obb.pt
   ```

### EfficientAd (anomalib)

1. Organize as imagens brutas:
   ```
   raw_dataset/
   ├── train/good/       imagens normais
   ├── test/good/        normais de teste
   └── test/defeito/     anômalas (opcional)
   ```
2. Gerar crops planificados pela pipeline YOLO + SAM:
   ```bash
   python src/data_prep/prepare_anomaly_crops.py --src raw_dataset --dst crops_dataset
   ```
3. Treinar:
   ```bash
   python src/train_anomalib.py --epochs 100
   ```
4. Copiar o checkpoint final:
   ```bash
   cp results/EfficientAd/etiqueta/latest/weights/lightning/model.ckpt models/anomalib.ckpt
   ```

**Observações:**
- Primeira execução do EfficientAd baixa ~1.5 GB do imagenette (usado no penalty loss). Só uma vez.
- EfficientAd exige `batch_size=1`. É limitação do modelo.
- No Windows, pode ser necessário `num_workers=0` (já é o padrão).

---

## Uso programático

```python
import torch
import cv2
from anomalib.models import EfficientAd

model = EfficientAd.load_from_checkpoint("models/anomalib.ckpt", map_location="cuda")
model.eval().cuda()

img = cv2.imread("meu_crop.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))
t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

with torch.no_grad():
    out = model(t)

print(f"Score: {out.pred_score.item():.3f}")
```

---

## Limitações conhecidas

- **Defeitos muito pequenos** (< 10 px na imagem original) ou de baixo contraste podem passar despercebidos pelo EfficientAd. Nesses casos, o ideal é usar PatchCore com features do `layer1` ou complementar com CV clássica (blob detection).
- **Etiquetas nunca vistas** no treino do YOLO podem não ser detectadas. Basta anotar mais exemplos e retreinar.
- **Pipeline exige** uma etiqueta razoavelmente reta na imagem (o SAM não conserta rotações extremas).
