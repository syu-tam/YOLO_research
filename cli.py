# cli.py

import argparse
import wandb
from models.yolo import YOLO
from pathlib import Path
import os

# -------------------------
# 引数で seed を受け取る
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
seed = args.seed

# 設定
mode = "train"
method = 'yolov11'
model_size = 'n'
data = "coco.yaml"
batch = 64
epochs = 500
patience = 50
is_wandb = True

# 保存先設定
train_project = os.path.join(f"result/v{method.replace('yolov', '')}/{data.replace('.yaml', '')}/{model_size}/LD(t=1)")
val_project = os.path.join(f"result/v{method.replace('yolov', '')}/{data.replace('.yaml', '')}/{model_size}/LD(t=1)")
name = f"{model_size}_seed{seed}"

# モデル設定ファイル
if 'yolo' in method:
    model_config = f"cfg/models/{method.replace('yolov', 'v')}/{method}{model_size}.yaml"
else:
    raise ValueError(f"無効なモデルの種類が指定されました: {method}")

if mode == 'train':
    if not is_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    wandb.init(project=f"{data.replace('.yaml', '')}", name=f"{method.replace('yolov', 'v')}_{model_size}_seed{seed}_LD(t=1)")

    if 'yolo' in method:
        model = YOLO(model_config, task='detect')

    model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        patience=patience,
        project=train_project,
        name=name,
        seed=seed
    )

elif mode == 'val':
    model_path = os.path.join(Path(train_project), Path(model_size), "weights/best.pt")
    print(f"Loading model from: {model_path}")

    model = YOLO(str(model_path), task='detect')

    model.val(
        data=data,
        batch=batch,
        project=val_project,
        name=name
    )
