# Ultralytics YOLO 🚀, AGPL-3.0 license

import math  # 数学関数を提供するモジュール
import random  # 乱数生成モジュール
from copy import copy  # オブジェクトのコピーを作成するためのモジュール

import numpy as np  # 数値計算ライブラリ
import torch.nn as nn  # ニューラルネットワークモジュール

from data import build_dataloader, build_yolo_dataset  # データローダーとデータセット構築関数をインポート
from engine.trainer import BaseTrainer  # トレーナーの基本クラスをインポート
from models import yolo  # yoloモデル関連のものをインポート
from nn.tasks import DetectionModel  # DetectionModelクラスをインポート
from utils import LOGGER, RANK  # ユーティリティ関数と定数をインポート
from utils.plotting import plot_images, plot_labels, plot_results  # プロット関連の関数をインポート
from utils.torch_utils import de_parallel, torch_distributed_zero_first  # 分散学習関連の関数をインポート


class DetectionTrainer(BaseTrainer):
    # 検出モデルに基づいてトレーニングするためのBaseTrainerクラスを拡張するクラス。
    # 例：
    #     ```python
    #     from models.yolo.detect import DetectionTrainer
    #
    #     args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
    #     trainer = DetectionTrainer(overrides=args)
    #     trainer.train()
    #     ```

    def build_dataset(self, img_path, mode="train", batch=None):
        # YOLOデータセットを構築します。
        #
        # 引数：
        #     img_path (str): 画像を含むフォルダへのパス。
        #     mode (str): 「train」モードまたは「val」モード。ユーザーは各モードに対して異なる拡張機能をカスタマイズできます。
        #     batch (int, optional): バッチのサイズ。「rect」用です。デフォルトはNoneです。
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)  # グリッドサイズを計算
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)  # YOLOデータセットを構築して返す

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # データローダーを構築して返します。
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."  # モードがtrainまたはvalであることを確認
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP。DDPの場合、データセットの*.cacheを一度だけ初期化
            dataset = self.build_dataset(dataset_path, mode, batch_size)  # データセットを構築
        shuffle = mode == "train"  # トレーニングモードの場合はシャッフル
        if getattr(dataset, "rect", False) and shuffle:  # データセットがrectでシャッフルの場合
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")  # 警告ログを出力
            shuffle = False  # シャッフルを無効化
        workers = self.args.workers if mode == "train" else self.args.workers * 2  # ワーカ数を設定
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # データローダーを構築して返す

    def preprocess_batch(self, batch):
        # 画像のバッチをスケーリングしてfloatに変換することにより、前処理します。
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255  # 画像をデバイスに転送して正規化
        if self.args.multi_scale:  # マルチスケールを使用する場合
            imgs = batch["img"]  # 画像を取得
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size。画像をリサイズするサイズを計算
            sf = sz / max(imgs.shape[2:])  # scale factor。スケールファクターを計算
            if sf != 1:  # スケールファクターが1でない場合
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)。新しい形状を計算
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)  # 画像をリサイズ
            batch["img"] = imgs  # 画像を更新
        return batch  # 更新されたバッチを返す

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model。モデルにクラス数をアタッチ
        self.model.names = self.data["names"]  # attach class names to model。モデルにクラス名をアタッチ
        self.model.args = self.args  # attach hyperparameters to model。モデルにハイパーパラメータをアタッチ
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        # YOLO検出モデルを返します。
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)  # 検出モデルを初期化
        if weights:  # 重みが指定されている場合
            model.load(weights)  # 重みをロード
        return model  # モデルを返す

    def get_validator(self):
        # YOLOモデル検証用のDetectionValidatorを返します。
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" , "  feature_loss" # 損失名を設定
        #self.loss_names = "box_loss", "cls_loss", "dfl_loss" 
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )  # DetectionValidatorを返す

    def label_loss_items(self, loss_items=None, prefix="train"):
        # ラベル付けされたトレーニング損失アイテムテンソルを含む損失dictを返します。
        # 分類には必要ありませんが、セグメンテーションと検出には必要です
        keys = [f"{prefix}/{x}" for x in self.loss_names]  # 損失名のリストを作成
        if loss_items is not None:  # 損失アイテムがNoneでない場合
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats。テンソルを5桁の浮動小数点数に変換
            return dict(zip(keys, loss_items))  # 損失辞書を返す
        else:  # 損失アイテムがNoneの場合
            return keys  # 損失名リストを返す

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        # エポック、GPUメモリ、損失、インスタンス、およびサイズでトレーニングの進行状況のフォーマットされた文字列を返します。
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (  # フォーマット文字列を作成
            "Epoch",  # エポック
            "GPU_mem",  # GPUメモリ
            *self.loss_names,  # 損失名
            "Instances",  # インスタンス数
            "Size",  # サイズ
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        # アノテーション付きのトレーニングサンプルをプロットします。
        plot_images(
            images=batch["img"],  # 画像データ
            batch_idx=batch["batch_idx"],  # バッチインデックス
            cls=batch["cls"].squeeze(-1),  # クラスラベル
            bboxes=batch["bboxes"],  # バウンディングボックス
            paths=batch["im_file"],  # 画像ファイルパス
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存ファイル名
            on_plot=self.on_plot,  # プロットコールバック関数
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        # CSVファイルからメトリックをプロットします。
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png。results.pngを保存

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        # YOLOモデルのラベル付きトレーニングプロットを作成します。
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)  # すべてのラベルのバウンディングボックスを結合
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)  # すべてのラベルのクラスラベルを結合
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)  # ラベルをプロット