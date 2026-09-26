# Scripts do projeto

Todos os scripts rodam a partir de `E:\yolo` no ambiente conda `yolo`.

## Índice

- [`train.py`](#trainpy) — treinar YOLO em um dataset
- [`predict_image.py`](#predict_imagepy) — testar em uma imagem
- [`predict_camera.py`](#predict_camerapy) — testar em tempo real com a webcam
- [`import_labelstudio.py`](#import_labelstudiopy) — importar export do Label Studio e organizar em train/val/test
- [`crop_detections.py`](#crop_detectionspy) — recortar detecções de um dataset inteiro em arquivos separados
- [`crop_video.py`](#crop_videopy) — cortar uma região fixa de um vídeo (ex: só a webcam)

---

## `train.py`

Treina um modelo YOLO26 em qualquer dataset organizado no formato Ultralytics.

**Uso mais simples** (treina em `my_dataset`):
```bash
python train.py
```

**Trocar dataset:**
```bash
python train.py bottledata
```

**Passar direto o data.yaml:**
```bash
python train.py bottledata/data.yaml
```

**Opções principais:**
| Flag | Padrão | Descrição |
|---|---|---|
| `dataset` | `my_dataset` | Pasta do dataset ou caminho do data.yaml |
| `--model` | `yolo26n.pt` | Modelo base ou .pt pra fine-tuning |
| `--epochs` | `100` | Número de épocas |
| `--imgsz` | `640` | Tamanho da imagem |
| `--batch` | `16` | Tamanho do batch |
| `--workers` | `4` | Workers do dataloader |
| `--patience` | `0` | Early stopping (0 = desligado) |
| `--name` | `exp-<dataset>` | Nome da run |
| `--device` | auto | `cpu`, `0`, `0,1`… |

**Exemplos:**
```bash
python train.py bottledata --epochs 50 --batch 8
python train.py bottledata --model yolo26s.pt
python train.py bottledata --model runs/detect/runs/train/exp-bottledata/weights/best.pt   # fine-tuning
python train.py bottledata --device cpu
```

**Saída:** os pesos ficam em `runs/detect/runs/train/exp-<dataset>/weights/best.pt`. No final imprime o Recall médio (R).

---

## `predict_image.py`

Roda o YOLO em uma imagem (ou vídeo, ou pasta) e salva o resultado anotado.

**Uso simples:**
```bash
python predict_image.py gato.jpg
```

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `image` | — | Caminho da imagem/vídeo/pasta |
| `--weights` | auto | `.pt` a usar (padrão: `best.pt` mais recente em `runs/`) |
| `--conf` | `0.25` | Confiança mínima |
| `--imgsz` | `640` | Tamanho da imagem |
| `--save-dir` | `runs/predict` | Pasta de saída |

**Exemplos:**
```bash
python predict_image.py "foto com espaco.jpg" --conf 0.5
python predict_image.py meu_video.mp4
python predict_image.py my_dataset/images/val    # roda em todas as imagens da pasta
python predict_image.py foto.jpg --weights runs/detect/runs/train/exp-bottledata/weights/best.pt
```

**Saída:** imagem/vídeo anotado em `runs/predict/img/` + terminal imprime classe, confiança e bbox de cada detecção.

---

## `predict_camera.py`

Abre a webcam com OpenCV e roda detecção YOLO em tempo real.

**Uso:**
```bash
python predict_camera.py
```

Pressione **`q`** para sair.

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `--weights` | auto | `.pt` a usar (padrão: `best.pt` mais recente) |
| `--source` | `0` | Índice da câmera |
| `--conf` | `0.70` | Confiança mínima |
| `--imgsz` | `640` | Tamanho da imagem |

**Exemplos:**
```bash
python predict_camera.py --source 1              # segunda câmera
python predict_camera.py --conf 0.5              # mais permissivo
python predict_camera.py --conf 0.85             # só detecções muito seguras
```

**Se der erro de câmera:** tente outro `--source` (0, 1, 2). Feche apps que estejam usando a webcam (Zoom, Teams, browser).

---

## `import_labelstudio.py`

Importa um export do Label Studio (formato **YOLO with images**, `.zip`) e distribui as fotos em `train/val/test`, tratando corretamente imagens de background.

**Uso simples:**
```bash
python import_labelstudio.py export.zip
```

Vai criar/preencher `my_dataset/images/{train,val,test}/` e `my_dataset/labels/{train,val,test}/`.

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `zip` | — | Caminho do export do Label Studio |
| `--dataset` | `my_dataset` | Pasta destino |
| `--train` | `0.7` | Proporção de treino |
| `--val` | `0.2` | Proporção de validação |
| `--test` | `0.1` | Proporção de teste |
| `--seed` | `42` | Semente da divisão |
| `--prefix` | `ls_` | Prefixo dos arquivos novos |
| `--clean` | off | Limpa os splits antes de importar |

**Exemplos:**
```bash
python import_labelstudio.py bottledata.zip --dataset bottledata --clean
python import_labelstudio.py export.zip --train 0.8 --val 0.1 --test 0.1
```

**O que faz:**
1. Extrai o zip.
2. Separa imagens anotadas de background (`.txt` vazio ou ausente).
3. Distribui cada grupo proporcionalmente entre train/val/test.
4. Cria `.txt` vazio pras imagens de background (formato correto YOLO).
5. Atualiza `data.yaml` com as classes do `classes.txt` do export.
6. Apaga caches (`.cache`) antigos.

---

## `crop_detections.py`

Roda o YOLO em todas as imagens de um dataset e salva cada objeto detectado como arquivo separado. Útil pra montar o dataset de "normais" pra anomaly detection.

**Uso simples:**
```bash
python crop_detections.py bottles_ok
```

Cria a pasta `bottles_ok/` com um arquivo por detecção, nome tipo `foto123_00_garrafa_c87.jpg` (c87 = 87% de confiança).

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `output` | — | Pasta destino dos recortes |
| `--dataset` | `bottledata` | Dataset a percorrer |
| `--source` | — | Alternativa: pasta livre em vez de dataset YOLO |
| `--weights` | auto | `.pt` (padrão: mais recente) |
| `--conf` | `0.70` | Confiança mínima |
| `--imgsz` | `640` | Tamanho da imagem |
| `--padding` | `0` | Pixels extras ao redor da bbox |
| `--min-size` | `32` | Descarta recortes menores que isto |
| `--per-class` | off | Separa em subpastas por classe |

**Exemplos:**
```bash
python crop_detections.py bottles_ok --dataset my_dataset
python crop_detections.py bottles_ok --conf 0.85 --padding 10
python crop_detections.py bottles_ok --per-class
python crop_detections.py bottles_ok --source "C:\Users\jhoni\Downloads\fotos"
```

---

## `crop_video.py`

Corta uma região retangular fixa de um vídeo (ex: manter só a webcam do canto). Sem áudio na saída.

**Uso interativo** (desenha o retângulo com o mouse no primeiro frame):
```bash
python crop_video.py "C:\Users\jhoni\Videos\NVIDIA\Desktop\video.mp4"
```

Depois:
- Arraste o mouse pra desenhar o retângulo
- **ENTER** ou **ESPAÇO** pra confirmar
- **C** pra cancelar

**Uso direto** (se já sabe as coordenadas `x,y,largura,altura`):
```bash
python crop_video.py video.mp4 --roi 1280,720,640,480
```

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `video` | — | Caminho do vídeo |
| `--output` | `<nome>_crop.mp4` | Caminho de saída |
| `--roi` | — | `x,y,w,h` (pula seleção interativa) |

---

## Fluxo típico de trabalho

1. **Anotar no Label Studio** → exportar como *YOLO with images* → `export.zip`.
2. **Importar e organizar:** `python import_labelstudio.py export.zip --dataset bottledata --clean`
3. **Treinar:** `python train.py bottledata`
4. **Testar em imagem:** `python predict_image.py foto.jpg`
5. **Testar na câmera:** `python predict_camera.py`
6. **Recortar objetos** (pra anomaly detection ou dataset auxiliar): `python crop_detections.py bottles_ok`

## Dicas gerais

- Caminhos com espaços: sempre entre aspas duplas.
- Todos os scripts que usam pesos pegam automaticamente o `best.pt` **mais recente** em `runs/`. Pra forçar outro, use `--weights`.
- Cada dataset gera uma run separada (`runs/detect/runs/train/exp-<nome>/`), então treinos diferentes não se sobrescrevem.
- Se aparecer erro CUDA / GPU: verifique com `python -c "import torch; print(torch.cuda.is_available())"`. Se `False`, o PyTorch foi instalado sem CUDA — reinstalar com `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.
