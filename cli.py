import wandb  # Weights & Biases（WandB）をインポート（ログ管理ツール）
from models.yolo import YOLO  # YOLOモデルのクラスをインポート
from pathlib import Path  # パス操作のためのPathライブラリをインポート
import os  # OS関連の操作のためのライブラリ

# モードの設定（"train"または"val"）
mode = "train"  # 現在のモード（学習用に設定）
method = 'yolov11'  # 使用するYOLOのバージョン
model_size = 'n'  # モデルサイズ（例：s, m, l など）
data = "VOC.yaml"  # データセット設定ファイル
batch = 64  # バッチサイズ
epochs = 1000  # 学習エポック数
patience = 100  # 学習の早期終了設定（0は無効）
is_wandb = True  # WandBを使用するかどうか

# 結果を保存するディレクトリの設定
train_project = os.path.join(f"result/v{method.replace('yolov', '')}/train/{model_size}")
val_project = os.path.join(f"result/v{method.replace('yolov', '')}/val/{model_size}")
name = f"{model_size}"  # モデル名

# モデルの設定
if 'yolo' in method:
    # 使用するYOLOモデルの設定ファイルを決定
    model_config = f"cfg/models/{method.replace('yolov', 'v')}/{method}{model_size}.yaml"
else:
    # 無効なモデルタイプが指定された場合はエラーを投げる
    raise ValueError(f"無効なモデルの種類が指定されました: {method}")

if mode == 'train':
    if not is_wandb:
        # WandBを使用しない場合、環境変数で無効化
        os.environ["WANDB_DISABLED"] = "true"
    
    # WandBの初期化（ログ管理用）
    wandb.init(project="YOLO", name=f"{method.replace('yolov', 'v')}_{model_size}")

    if 'yolo' in method:
        # YOLOモデルを初期化（物体検出タスク用）
        model = YOLO(model_config, task='detect')
        
    # モデルのトレーニングを開始
    model.train(
        data=data,  # 使用するデータセット
        epochs=epochs,  # 学習エポック数
        batch=batch,  # バッチサイズ
        patience=patience,  # 早期終了設定
        project=train_project,  # 結果を保存するディレクトリ
        name=name  # モデル名
    )

elif mode == 'val':
    # 評価モードの場合、学習済みモデルのロード
    model_path = os.path.join(Path(train_project), Path(model_size), "weights/best.pt")
    print(f"Loading model from: {model_path}")
    
    # 保存された学習済みモデルをロード
    model = YOLO(str(model_path), task='detect')

    # モデルの評価を実行
    model.val(
        data=data,  # 評価データセット
        batch=batch,  # バッチサイズ
        project=val_project,  # 評価結果の保存ディレクトリ
        name=name  # モデル名
    )
