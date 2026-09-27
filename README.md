# Scripts do projeto

Pipeline focado em detecção de garrafas com **YOLO-OBB** (oriented bounding boxes).

Todos os scripts rodam a partir de `E:\yolo` no ambiente conda `yolo`.

## Índice

- [`import_labelstudio.py`](#import_labelstudiopy) — importar export do Label Studio e organizar em train/val/test
- [`train_obb.py`](#train_obbpy) — treinar YOLO-OBB em um dataset
- [`predict_obb.py`](#predict_obbpy) — testar em imagem/pasta/vídeo
- [`predict_camera_obb.py`](#predict_camera_obbpy) — testar em tempo real com webcam
- [`track_camera_obb.py`](#track_camera_obbpy) — tracking OBB em tempo real com contagem
- [`bytetrack_custom.yaml`](#bytetrack_customyaml) — config do tracker (persistência de ID)

---

## `import_labelstudio.py`

Importa um export do Label Studio (**YOLOv8 OBB with Images**, `.zip`) e distribui as fotos em `train/val/test`, tratando corretamente imagens de background.

**Uso simples:**
```bash
python import_labelstudio.py export.zip --dataset bottledata_obb --clean
```

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

**O que faz:**
1. Extrai o zip.
2. Separa imagens anotadas de background (`.txt` vazio ou ausente).
3. Distribui cada grupo proporcionalmente entre train/val/test.
4. Cria `.txt` vazio pras imagens de background (formato correto YOLO).
5. Atualiza `data.yaml` com as classes do `classes.txt` do export e caminho absoluto do dataset.
6. Apaga caches (`.cache`) antigos.

**No Label Studio, ao exportar escolha:** `YOLOv8 OBB with Images` (é o formato certo pro OBB).

---

## `train_obb.py`

Treina YOLO-OBB (bboxes rotacionadas).

**Uso simples:**
```bash
python train_obb.py bottledata_obb
```

Ou passando o `data.yaml` direto:
```bash
python train_obb.py bottledata_obb/data.yaml
```

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `dataset` | `bottledata_obb` | Pasta do dataset ou `data.yaml` |
| `--model` | `yolo26n-obb.pt` | Modelo base ou .pt pra fine-tuning |
| `--epochs` | `100` | Número de épocas |
| `--imgsz` | `640` | Tamanho da imagem |
| `--batch` | `16` | Tamanho do batch |
| `--workers` | `4` | Workers do dataloader |
| `--patience` | `0` | Early stopping (0 = desligado) |
| `--name` | `obb-<dataset>` | Nome da run |
| `--device` | auto | `cpu`, `0`, `0,1`… |

**Exemplos:**
```bash
python train_obb.py bottledata_obb --epochs 50 --batch 8
python train_obb.py bottledata_obb --model yolo26s-obb.pt
python train_obb.py bottledata_obb --model runs/obb/obb-bottledata_obb/weights/best.pt   # fine-tuning
python train_obb.py bottledata_obb --device cpu
```

**Saída:** pesos em `runs/obb/obb-<dataset>/weights/best.pt`. No final imprime o Recall médio (R).

---

## `predict_obb.py`

Predict OBB em imagem, pasta ou vídeo. Salva anotação e imprime detecções com ângulo.

**Uso:**
```bash
python predict_obb.py foto.jpg
python predict_obb.py foto.jpg --conf 0.7
python predict_obb.py pasta_com_fotos
python predict_obb.py video.mp4
```

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `source` | — | Imagem, pasta ou vídeo |
| `--weights` | auto | `.pt` OBB (padrão: `best.pt` mais recente em `runs/obb/`) |
| `--conf` | `0.50` | Confiança mínima |
| `--imgsz` | `640` | Tamanho da imagem |
| `--save-dir` | `runs/obb_predict` | Pasta de saída |

**Saída:** imagem/vídeo anotado em `runs/obb_predict/obb/` + terminal imprime cada detecção com classe, confiança, centro, tamanho e **ângulo em graus**.

---

## `predict_camera_obb.py`

Detecção OBB em tempo real usando webcam. Pressione **`q`** pra sair.

**Uso:**
```bash
python predict_camera_obb.py
```

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `--weights` | auto | `.pt` OBB (padrão: `best.pt` mais recente em `runs/obb/`) |
| `--source` | `0` | Índice da câmera |
| `--conf` | `0.50` | Confiança mínima |
| `--imgsz` | `640` | Tamanho da imagem no modelo |
| `--cam-w` | `1280` | Largura da captura da câmera |
| `--cam-h` | `720` | Altura da captura da câmera |
| `--window-w` | `1280` | Largura da janela de exibição |
| `--window-h` | `720` | Altura da janela de exibição |

**Exemplos:**
```bash
python predict_camera_obb.py --conf 0.7
python predict_camera_obb.py --source 1
python predict_camera_obb.py --window-w 1920 --window-h 1080
```

---

## `track_camera_obb.py`

Detecção OBB **com tracking** (IDs persistentes) e contagem em tempo real. Ideal pra saber quantas garrafas passaram.

**Uso:**
```bash
python track_camera_obb.py
```

**Opções:**
| Flag | Padrão | Descrição |
|---|---|---|
| `--weights` | auto | `.pt` OBB |
| `--source` | `0` | Câmera (int) ou caminho de vídeo |
| `--conf` | `0.60` | Confiança mínima |
| `--imgsz` | `640` | Tamanho da imagem |
| `--tracker` | `bytetrack_custom.yaml` | Config do tracker |
| `--save` | — | Salva vídeo anotado neste caminho |
| `--persist-frames` | `150` | Frames sem detecção pra contar como "passou" (~5s a 30fps) |
| `--cam-w` / `--cam-h` | `1280` / `720` | Resolução da câmera |
| `--window-w` / `--window-h` | `1280` / `720` | Tamanho da janela |

**Exemplos:**
```bash
python track_camera_obb.py                                        # webcam
python track_camera_obb.py --source video.mp4                     # vídeo
python track_camera_obb.py --source video.mp4 --save saida.mp4    # salvando
python track_camera_obb.py --tracker botsort.yaml                 # BoT-SORT (mais robusto)
```

**HUD (canto superior esquerdo, em amarelo):**
- `Na tela` — quantas garrafas o modelo vê agora.
- `Total vistas` — quantos IDs únicos apareceram desde o início.
- `Passaram` — quantas apareceram e saíram (após `--persist-frames` sem redetecção).

Pressione **`q`** pra sair. No fim imprime o resumo total.

---

## `bytetrack_custom.yaml`

Config do ByteTrack customizado — é o padrão do Ultralytics com **track_buffer aumentado** (30 → 300 frames ≈ 10s), pra segurar os IDs mais tempo sem detecção e evitar que a mesma garrafa vire vários IDs.

Editar se quiser ajustar:
- `track_buffer` — quantos frames aguenta sem detectar antes de matar o ID.
- `new_track_thresh` — quão confiante uma detecção precisa ser pra criar ID novo.
- `match_thresh` — tolerância pra reassociar bbox entre frames.

Alternativas prontas do Ultralytics:
- `bytetrack.yaml` — padrão, mais responsivo.
- `botsort.yaml` — mais robusto, com re-identificação, mas ~2x mais lento.

---

## Fluxo típico de trabalho

1. **Anotar no Label Studio** (usando **Rotated Rectangle**, `canRotate="true"` no template) → exportar como **YOLOv8 OBB with Images** → `export.zip`.
2. **Importar e organizar:** `python import_labelstudio.py export.zip --dataset bottledata_obb --clean`
3. **Treinar:** `python train_obb.py bottledata_obb`
4. **Testar em imagem:** `python predict_obb.py gato.jpg`
5. **Testar na câmera:** `python predict_camera_obb.py`
6. **Contar garrafas passando:** `python track_camera_obb.py`

## Dicas gerais

- Caminhos com espaços: sempre entre aspas duplas.
- Todos os scripts pegam automaticamente o `best.pt` **mais recente** em `runs/obb/`. Pra forçar outro, use `--weights`.
- Cada dataset gera uma run separada (`runs/obb/obb-<nome>/`), então treinos diferentes não se sobrescrevem.
- Se aparecer erro CUDA / GPU: verifique com `python -c "import torch; print(torch.cuda.is_available())"`. Se `False`, reinstale PyTorch com CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.
- Pra rodar em máquina sem GPU: passe `--device cpu` no `train_obb.py`. Nos scripts de predição, dá pra setar `set CUDA_VISIBLE_DEVICES=` antes de rodar (Windows/CMD) ou `$env:CUDA_VISIBLE_DEVICES=""` (PowerShell) — cai automaticamente pra CPU.
- Pra máquinas Intel sem GPU, exportar o modelo pra OpenVINO acelera 2-3x: `YOLO("best.pt").export(format="openvino", int8=True)` e depois usar `--weights best_openvino_model/`.
