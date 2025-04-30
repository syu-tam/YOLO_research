import argparse
import importlib
import traceback
from tqdm import tqdm
from pathlib import Path
import yaml
import sys
from types import SimpleNamespace

# --- 引数パース ---
p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="data.yaml ファイルパス")
p.add_argument("--img", type=int, default=640)
p.add_argument("--max", type=int, default=1)
args = p.parse_args()

# --- リポジトリルート設定 ---
REPO = Path(__file__).resolve().parent
sys.path.append(str(REPO))

# --- data.yaml をロードして Namespaceに変換 ---
with open(args.data, "r") as f:
    cfg_dict = yaml.safe_load(f)
# operator: 必須属性をすべて設定
cfg_dict["imgsz"]    = args.img
cfg_dict.setdefault("rect",    False)
cfg_dict.setdefault("cache",   False)
cfg_dict["single_cls"] = False      # ← これを追加：
cfg_dict.setdefault("task",    "train")
cfg_dict.setdefault("classes", None)
cfg_dict.setdefault("fraction", 1.0)
# optional: single_clsなど必要なら
# cfg_dict.setdefault("single_cls", False)

cfg = SimpleNamespace(**cfg_dict)

# --- build_yolo_dataset 読み込み ---
db = importlib.import_module("data.build")

# --- 画像フォルダへの絶対パスを組み立て ---
# cfg.path: データセットルート（yaml内指定）
root = (REPO / cfg.path).resolve()
# cfg.train: train画像ディレクトリ（yaml内指定）
img_path = str((root / cfg.train).resolve())

# --- Dataset作成（batch=1, worker=0, augmentあり） ---
dataset = db.build_yolo_dataset(
    cfg=cfg,
    data =cfg,
    img_path=img_path,   # ←ここにimg_pathを渡す
    batch=1,
    rect=False,
    stride=32
)

print(f"Dataset length: {len(dataset)}")

# --- 問題サンプルスキャン ---
bad, limit = [], max(args.max, 1)
for i in tqdm(range(len(dataset)), desc="scan"):
    try:
        _ = dataset[i]
    except Exception as e:
        bad.append((i, dataset.im_files[i], traceback.format_exc().splitlines()[-1]))
        if len(bad) >= limit:
            break

# --- 結果表示 ---
if not bad:
    print("🎉 エラーは見つかりませんでした。")
else:
    print("\n===== 問題サンプル =====")
    for idx, path, msg in bad:
        print(f"[{idx:>6}] {path}\n      {msg}\n" + "-"*60)
    print(f"↑ {len(bad)} 件のエラーを検出しました。")
