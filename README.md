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

**Com GPU NVIDIA (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Sem GPU NVIDIA (CPU ou iGPU Intel):**
```bash
pip install -r requirements.txt
```

O `requirements.txt` já traz `openvino` e `onnx` — necessários pra rodar o YOLO na iGPU Intel.

### Passo 2 — verificar

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import openvino as ov; print('OpenVINO devices:', ov.Core().available_devices)"
```

Se aparecer `GPU` (ou `GPU.0`) na lista do OpenVINO, sua iGPU Intel está pronta pra uso. Se aparecer só `CPU`, atualize o driver da Intel UHD/Iris pelo site da Intel — o OpenVINO usa o runtime OpenCL que vem junto.

---

## Escolher onde rodar (CPU / iGPU Intel / GPU NVIDIA)

Todos os scripts de inferência aceitam `--device`. Valores possíveis:

| `--device`  | Onde roda o YOLO                | Onde roda o SAM       | Requisitos                                             |
|-------------|---------------------------------|-----------------------|--------------------------------------------------------|
| `cpu`       | CPU                             | CPU                   | Nenhum (funciona em qualquer PC)                       |
| `intel`     | iGPU Intel via OpenVINO         | CPU (fallback)        | Modelo exportado pra OpenVINO + driver Intel atual     |
| `intel:npu` | NPU Intel (se houver)           | CPU (fallback)        | CPU Intel Core Ultra + driver da NPU                   |
| `nvidia`    | GPU NVIDIA (CUDA)               | GPU NVIDIA (CUDA)     | PyTorch instalado com CUDA + placa NVIDIA              |

> **Por que o SAM não vai pra iGPU Intel?** O `SamRefiner` usa o SAM em PyTorch puro, e PyTorch não tem backend pra Intel iGPU. Ele é fixado em CPU quando você escolhe `--device intel`. Se precisar de SAM acelerado, use uma GPU NVIDIA com `--device nvidia`.

### Passo obrigatório antes de usar `--device intel`

Exportar o YOLO pra OpenVINO **uma vez só**:

```bash
python src/export_openvino.py --imgsz 640
```

Isso gera `models/yolo_obb_openvino_model/`. Se essa pasta existir, o script de câmera passa a usá-la automaticamente (imprime `[OpenVINO]` no cabeçalho).

Pra remover e voltar a usar só o `.pt`:

```bash
rmdir /s /q models\yolo_obb_openvino_model     # Windows
rm -rf models/yolo_obb_openvino_model          # Linux/macOS
```

### Exemplos

```bash
# CPU (padrão) — funciona em qualquer PC
python src/predict_camera.py

# iGPU Intel — YOLO na iGPU, SAM em CPU
python src/predict_camera.py --device intel
python src/predict_camera.py --device intel --sam --sam-model models/mobile_sam.pt

# GPU NVIDIA — YOLO e SAM na GPU
python src/predict_camera.py --device nvidia
python src/predict_camera.py --device nvidia --sam --sam-model models/mobile_sam.pt
```

Ao rodar, o script imprime qual device ele está de fato usando:

```
Pesos     : models\yolo_obb_openvino_model  [OpenVINO]
YOLO device: intel:gpu
SAM device : cpu  (SAM só roda em CPU ou CUDA)
```

### Troubleshooting

- **"Device with 'GPU' name is not registered" com `--device intel`** → driver Intel desatualizado. Baixe o driver mais novo da Intel pra sua UHD/Iris/Arc.
- **`--device nvidia` cai em CPU silenciosamente** → PyTorch está instalado sem CUDA. Reinstale seguindo o Passo 1 acima.
- **SAM ficando amarelo (nunca verde) com `--device intel`** → o script já trata isso; se ainda acontecer, olhe se apareceu `[SAM] falhou: ...` no terminal e me mande a mensagem.
- **Ganho pequeno na iGPU** → a Intel UHD é fraca; o ganho maior costuma vir de reduzir `--imgsz` (ex. `--imgsz 480`) e aumentar `--refine-every` (ex. `10`).

---

## Como usar (modelos já treinados)

Todos os scripts aceitam `--device cpu | intel | nvidia`. Veja a seção **Escolher onde rodar** acima pros detalhes.

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
