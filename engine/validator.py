import json
import time
from pathlib import Path

import numpy as np
import torch

from cfg import get_cfg, get_save_dir
from data.utils import check_cls_dataset, check_det_dataset
from nn.autobackend import AutoBackend
from utils import LOGGER, TQDM, callbacks, colorstr, emojis
from utils.checks import check_imgsz
from utils.ops import Profile
from utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    # BaseValidator.
    #
    # バリデーターを作成するための基本クラス。
    #
    # 属性：
    #     args (SimpleNamespace): バリデーターの構成。
    #     dataloader (DataLoader): 検証に使用するデータローダー。
    #     pbar (tqdm): 検証中に更新するプログレスバー。
    #     model (nn.Module): 検証するモデル。
    #     data (dict): データ辞書。
    #     device (torch.device): 検証に使用するデバイス。
    #     batch_i (int): 現在のバッチインデックス。
    #     training (bool): モデルがトレーニングモードかどうか。
    #     names (dict): クラス名。
    #     seen: 検証中にこれまでに見た画像の数を記録します。
    #     stats: 検証中の統計のプレースホルダー。
    #     confusion_matrix: 混同行列のプレースホルダー。
    #     nc: クラス数。
    #     iouv: (torch.Tensor): 0.05の間隔で0.50〜0.95のIoUしきい値。
    #     jdict (dict): JSON検証結果を保存する辞書。
    #     speed (dict): キー「preprocess」、「inference」、「loss」、「postprocess」、およびそれぞれのバッチ処理時間（ミリ秒単位）を含む辞書。
    #     save_dir (Path): 結果を保存するディレクトリ。
    #     plots (dict): 可視化のためにプロットを保存する辞書。
    #     callbacks (dict): さまざまなコールバック関数を保存する辞書。

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        # BaseValidatorインスタンスを初期化します。
        #
        # 引数：
        #     dataloader (torch.utils.data.DataLoader): 検証に使用するデータローダー。
        #     save_dir (Path, optional): 結果を保存するディレクトリ（オプション）。
        #     pbar (tqdm.tqdm): 進行状況を表示するためのプログレスバー。
        #     args (SimpleNamespace): バリデーターの構成。
        #     _callbacks (dict): さまざまなコールバック関数を保存する辞書。
        self.args = get_cfg(overrides=args)  # 設定を取得
        self.dataloader = dataloader  # データローダーを設定
        self.pbar = pbar  # プログレスバーを設定
        self.stride = None  # ストライドを設定
        self.data = None  # データを設定
        self.device = None  # デバイスを設定
        self.batch_i = None  # バッチインデックスを設定
        self.training = True  # トレーニングモードを設定
        self.names = None  # 名前を設定
        self.seen = None  # 参照を設定
        self.stats = None  # 統計を設定
        self.confusion_matrix = None  # 混同行列を設定
        self.nc = None  # クラス数を設定
        self.iouv = None  # IOUしきい値を設定
        self.jdict = None  # JSON辞書を設定
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # 速度を設定

        self.save_dir = save_dir or get_save_dir(self.args)  # 保存ディレクトリを設定
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # ディレクトリを作成
        if self.args.conf is None:  # 信頼度がNoneの場合
            self.args.conf = 0.001  # default conf=0.001。信頼度を設定
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)  # 画像サイズをチェック

        self.plots = {}  # プロットを設定
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # コールバックを設定

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        # 検証プロセスを実行し、データローダーで推論を実行し、パフォーマンスメトリックを計算します。
        self.training = trainer is not None  # トレーニングモードを設定
        augment = self.args.augment and (not self.training)  # 拡張を設定
        if self.training:  # トレーニングモードの場合
            self.device = trainer.device  # デバイスを設定
            self.data = trainer.data  # データを設定
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp  # 半精度を設定
            model = trainer.ema.ema or trainer.model  # モデルを設定
            model = model.half() if self.args.half else model.float()  # 半精度または浮動小数点に変換
            if hasattr(model.model[-1], 'current_tal_topk'):  # tal_topkがある場合
                current_tal_topk = model.model[-1].current_tal_topk  # current_tal_topkを取得
                model.model[-1].current_tal_topk = current_tal_topk  # current_tal_topkを設定
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)  # 損失を初期化
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)  # プロットを設定
            model.eval()  # 評価モードを設定
        else:  # トレーニングモードでない場合
            if str(self.args.model).endswith(".yaml"):  # モデルがyamlで終わる場合
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")  # 警告ログを出力
            callbacks.add_integration_callbacks(self)  # 統合コールバックを追加
            model = AutoBackend(  # AutoBackendを初期化
                weights=model or self.args.model,  # 重みを設定
                device=select_device(self.args.device, self.args.batch),  # デバイスを設定
                dnn=self.args.dnn,  # DNNを設定
                data=self.args.data,  # データを設定
                fp16=self.args.half,  # 半精度を設定
            )
            # self.model = model
            self.device = model.device  # update device。デバイスを更新
            self.args.half = model.fp16  # update half。半精度を更新
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine  # 属性を取得
            imgsz = check_imgsz(self.args.imgsz, stride=stride)  # 画像サイズをチェック
            if engine:  # エンジンがある場合
                self.args.batch = model.batch_size  # バッチサイズを設定
            elif not pt and not jit:  # ptとjitがない場合
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1。デフォルトのバッチサイズ1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")  # ログを出力

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:  # データ形式がyamlの場合
                self.data = check_det_dataset(self.args.data)  # 検出データセットをチェック
            elif self.args.task == "classify":  # 分類タスクの場合
                self.data = check_cls_dataset(self.args.data, split=self.args.split)  # 分類データセットをチェック
            else:  # その他の場合
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))  # エラーを発生

            if self.device.type in {"cpu", "mps"}:  # デバイスタイプがcpuまたはmpsの場合
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading。ワーカーを0に設定
            if not pt:  # ptがない場合
                self.args.rect = False  # アスペクト比固定を無効化
            self.stride = model.stride  # used in get_dataloader() for padding。パディングに使用されるストライド
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)  # データローダーを取得

            model.eval()  # 評価モードを設定
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup。ウォームアップ

        self.run_callbacks("on_val_start")  # 検証開始時にコールバックを実行
        dt = (  # 時間計測器を初期化
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))  # プログレスバーを初期化
        self.init_metrics(de_parallel(model))  # メトリクスを初期化
        self.jdict = []  # empty before each val。各検証前に空にする
        for batch_i, batch in enumerate(bar):  # バッチを反復処理
            self.run_callbacks("on_val_batch_start")  # 検証バッチの開始時にコールバックを実行
            self.batch_i = batch_i  # バッチインデックスを設定
            # Preprocess
            with dt[0]:  # 前処理時間
                batch = self.preprocess(batch)  # バッチを前処理

            # Inference
            with dt[1]:  # 推論時間
                preds = model(batch["img"], augment=augment)  # モデルを適用

            # Loss
            with dt[2]:  # 損失時間
                if self.training:  # トレーニングモードの場合
                    self.loss += model.loss(batch, preds)[1]  # 損失を計算

            # Postprocess
            with dt[3]:  # 後処理時間
                preds = self.postprocess(preds)  # 予測を後処理

            self.update_metrics(preds, batch)  # メトリクスを更新
            if self.args.plots and batch_i < 3:  # プロットする場合
                self.plot_val_samples(batch, batch_i)  # 検証サンプルをプロット
                self.plot_predictions(batch, preds, batch_i)  # 予測をプロット

            self.run_callbacks("on_val_batch_end")  # 検証バッチの終了時にコールバックを実行
        stats = self.get_stats()  # 統計を取得
        self.check_stats(stats)  # 統計をチェック
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))  # 速度を計算
        self.finalize_metrics()  # メトリクスを確定
        self.print_results()  # 結果を出力
        self.run_callbacks("on_val_end")  # 検証終了時にコールバックを実行
        if self.training:  # トレーニングモードの場合
            model.float()  # モデルを浮動小数点数に変換
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}  # 結果を取得
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats。5桁の浮動小数点数として結果を返す
        else:  # トレーニングモードでない場合
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )  # ログを出力
            if self.args.save_json and self.jdict:  # JSONを保存する場合
                with open(str(self.save_dir / "predictions.json"), "w") as f:  # ファイルを開く
                    LOGGER.info(f"Saving {f.name}...")  # ログを出力
                    json.dump(self.jdict, f)  # flatten and save。平坦化して保存
                stats = self.eval_json(stats)  # update stats。統計を更新
            if self.args.plots or self.args.save_json:  # プロットする場合
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")  # ログを出力
            return stats  # 統計を返す

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        # IoUを使用して、予測をグランドトゥルースオブジェクト（pred_classes、true_classes）と一致させます。
        #
        # 引数：
        #     pred_classes (torch.Tensor): 形状(N,)の予測されたクラスインデックス。
        #     true_classes (torch.Tensor): 形状(M,)のターゲットクラスインデックス。
        #     iou (torch.Tensor): 予測と真実のグラウンドのペアワイズIoU値を含むNxMテンソル
        #     use_scipy (bool): マッチングにscipyを使用するかどうか（より正確）。
        #
        # 戻り値：
        #     (torch.Tensor): 10個のIoUしきい値に対する形状(N,10)の正しいテンソル。
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)  # 正しい行列を初期化
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes  # 正しいクラス
        iou = iou * correct_class  # zero out the wrong classes。間違ったクラスをゼロにする
        iou = iou.cpu().numpy()  # GPUからCPUに移動
        for i, threshold in enumerate(self.iouv.cpu().tolist()):  # 各しきい値を反復処理
            if use_scipy:  # scipyを使用する場合
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands。すべてのコマンドでインポートしないようにスコープインポート

                cost_matrix = iou * (iou >= threshold)  # コスト行列を計算
                if cost_matrix.any():  # コスト行列に何かがある場合
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)  # 線形割り当て
                    valid = cost_matrix[labels_idx, detections_idx] > 0  # 有効な割り当て
                    if valid.any():  # 有効な割り当てがある場合
                        correct[detections_idx[valid], i] = True  # 正しい割り当てをマーク
            else:  # scipyを使用しない場合
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match。IoU>しきい値およびクラスの一致
                matches = np.array(matches).T  # 変換
                if matches.shape[0]:  # 一致するものがある場合
                    if matches.shape[0] > 1:  # 複数の一致がある場合
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]  # 並べ替え
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 重複を削除
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 重複を削除
                    correct[matches[:, 1].astype(int), i] = True  # 正しい割り当てをマーク
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)  # 正しいテンソルを返す

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        # 指定されたコールバックを追加します。
        self.callbacks[event].append(callback)  # コールバックを追加

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        # 特定のイベントに関連付けられているすべてのコールバックを実行します。
        for callback in self.callbacks.get(event, []):  # イベントのコールバックを反復処理
            callback(self)  # コールバックを実行

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        # データセットパスとバッチサイズからデータローダーを取得します。
        raise NotImplementedError("get_dataloader function not implemented for this validator")  # エラーを発生

    def build_dataset(self, img_path):
        """Build dataset."""
        # データセットを構築します。
        raise NotImplementedError("build_dataset function not implemented in validator")  # エラーを発生

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        # 入力バッチを前処理します。
        return batch  # バッチを返す

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        # 予測を前処理します。
        return preds  # 予測を返す

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        # YOLOモデルのパフォーマンスメトリクスを初期化します。
        pass  # 何もしない

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        # 予測とバッチに基づいてメトリクスを更新します。
        pass  # 何もしない

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        # すべてのメトリクスを確定して返します。
        pass  # 何もしない

    def get_stats(self):
        """Returns statistics about the model's performance."""
        # モデルのパフォーマンスに関する統計を返します。
        return {}  # 空の辞書を返す

    def check_stats(self, stats):
        """Checks statistics."""
        # 統計をチェックします。
        pass  # 何もしない

    def print_results(self):
        """Prints the results of the model's predictions."""
        # モデルの予測の結果を出力します。
        pass  # 何もしない

    def get_desc(self):
        """Get description of the YOLO model."""
        # YOLOモデルの説明を取得します。
        pass  # 何もしない

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        # YOLOトレーニング/検証で使用されるメトリックキーを返します。
        return []  # 空のリストを返す

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        # プロットを登録します（例：コールバックで使用するため）。
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}  # プロットを登録

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        # トレーニング中に検証サンプルをプロットします。
        pass  # 何もしない

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        # バッチ画像にYOLOモデルの予測をプロットします。
        pass  # 何もしない

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        # 予測をJSON形式に変換します。
        pass  # 何もしない

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        # 予測統計のJSON形式を評価して返します。
        pass  # 何もしない