# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from data import build_dataloader, build_yolo_dataset, converter
from engine.validator import BaseValidator
from utils import LOGGER, ops
from utils.checks import check_requirements
from utils.ops import non_max_suppression
from utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    # 検出モデルに基づいて検証するためのBaseValidatorクラスを拡張するクラス。
    #
    # 例:
    #     ```python
    #     from models.yolo.detect import DetectionValidator
    #
    #     args = dict(model="yolo11n.pt", data="coco8.yaml")
    #     validator = DetectionValidator(args=args)
    #     validator()
    #     ```

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        # 必要な変数と設定を使用して検出モデルを初期化します。
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)  # 親クラスを初期化
        self.nt_per_class = None  # クラスごとのターゲット数
        self.nt_per_image = None  # 画像ごとのターゲット数
        self.is_coco = False  # COCOデータセットかどうか
        self.is_lvis = False  # LVISデータセットかどうか
        self.class_map = None  # クラスマップ
        self.args.task = "detect"  # タスクを検出に設定
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)  # 検出メトリクスを初期化
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95。mAP@0.5:0.95のIoUベクトル
        self.niou = self.iouv.numel()  # IoUベクトル数
        self.lb = []  # for autolabelling。自動ラベル付け用
        if self.args.save_hybrid:  # ハイブリッド保存がTrueの場合
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )  # 警告ログを出力

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        # YOLOトレーニング用に画像のバッチを前処理します。
        batch["img"] = batch["img"].to(self.device, non_blocking=True)  # 画像をデバイスに移動
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255  # 画像をfloatに変換して正規化
        for k in ["batch_idx", "cls", "bboxes"]:  # キーを反復処理
            batch[k] = batch[k].to(self.device)  # デバイスに移動

        if self.args.save_hybrid:  # ハイブリッド保存がTrueの場合
            height, width = batch["img"].shape[2:]  # 高さ、幅
            nb = len(batch["img"])  # バッチサイズ
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)  # バウンディングボックスをスケーリング
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]  # ラベルを生成

        return batch  # バッチを返す

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        # YOLOの評価メトリクスを初期化します。
        val = self.data.get(self.args.split, "")  # validation path。検証パス
        self.is_coco = (  # COCOデータセットかどうか
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS。LVISデータセットかどうか
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))  # クラスマップを初期化
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val。最終検証を実行
        self.names = model.names  # 名前を設定
        self.nc = len(model.names)  # クラス数を設定
        self.metrics.names = self.names  # メトリクスの名前を設定
        self.metrics.plot = self.args.plots  # メトリクスのプロットを設定
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)  # 混同行列を初期化
        self.seen = 0  # シーンを0に設定
        self.jdict = []  # JDictを初期化
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])  # 統計を初期化

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        # YOLOモデルのクラスメトリクスを要約する書式設定された文字列を返します。
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")  # 文字列を返す

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        # 非最大抑制を予測出力に適用します。

        return non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )  # NMSを適用

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        # 検証用に画像のバッチとアノテーションを準備します。
        idx = batch["batch_idx"] == si  # バッチインデックスをチェック
        cls = batch["cls"][idx].squeeze(-1)  # クラス
        bbox = batch["bboxes"][idx]  # バウンディングボックス
        ori_shape = batch["ori_shape"][si]  # 元の形状
        imgsz = batch["img"].shape[2:]  # 画像サイズ
        ratio_pad = batch["ratio_pad"][si]  # 割合とパディング
        if len(cls):  # クラスがある場合
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes。ターゲットボックス
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels。ネイティブスペースラベル
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}  # 辞書を返す

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        # 検証用に画像のバッチとアノテーションを準備します。
        predn = pred.clone()  # 複製を作成
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred。ネイティブスペース予測
        return predn  # 複製を返す

    def update_metrics(self, preds, batch):
        """Metrics."""
        # メトリクス。
        for si, pred in enumerate(preds):  # 予測を反復処理
            self.seen += 1  # 参照をインクリメント
            npr = len(pred)  # 予測数
            stat = dict(  # 統計を初期化
                conf=torch.zeros(0, device=self.device),  # 信頼度
                pred_cls=torch.zeros(0, device=self.device),  # 予測クラス
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 真陽性
            )
            pbatch = self._prepare_batch(si, batch)  # バッチを準備
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")  # クラスとバウンディングボックスを取得
            nl = len(cls)  # ラベル数
            stat["target_cls"] = cls  # ターゲットクラスを設定
            stat["target_img"] = cls.unique()  # ターゲット画像を設定
            if npr == 0:  # 予測がない場合
                if nl:  # ラベルがある場合
                    for k in self.stats.keys():  # 統計を反復処理
                        self.stats[k].append(stat[k])  # 統計を追加
                    if self.args.plots:  # プロットがTrueの場合
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)  # 混同行列を処理
                continue  # 次の画像へ

            # Predictions
            if self.args.single_cls:  # シングルクラスの場合
                pred[:, 5] = 0  # クラスを0に設定
            predn = self._prepare_pred(pred, pbatch)  # 予測を準備
            stat["conf"] = predn[:, 4]  # 信頼度を設定
            stat["pred_cls"] = predn[:, 5]  # 予測クラスを設定

            # Evaluate
            if nl:  # ラベルがある場合
                stat["tp"] = self._process_batch(predn, bbox, cls)  # 真陽性を処理
                if self.args.plots:  # プロットがTrueの場合
                    self.confusion_matrix.process_batch(predn, bbox, cls)  # 混同行列を処理
            for k in self.stats.keys():  # 統計を反復処理
                self.stats[k].append(stat[k])  # 統計を追加

            # Save
            if self.args.save_json:  # JSONを保存する場合
                self.pred_to_json(predn, batch["im_file"][si])  # 予測をJSONに変換
            if self.args.save_txt:  # テキストを保存する場合
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )  # テキストを保存

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        # メトリクスの速度と混同行列の最終値を設定します。
        self.metrics.speed = self.speed  # 速度を設定
        self.metrics.confusion_matrix = self.confusion_matrix  # 混同行列を設定

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        # メトリクスの統計と結果辞書を返します。
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy。numpyに変換
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)  # クラスごとのターゲット数
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)  # 画像ごとのターゲット数
        stats.pop("target_img", None)  # ターゲット画像を削除
        if len(stats) and stats["tp"].any():  # 統計があり、真陽性がある場合
            self.metrics.process(**stats)  # メトリクスを処理
        return self.metrics.results_dict  # 結果辞書を返す

    def print_results(self):
        """Prints training/validation set metrics per class."""
        # クラスごとのトレーニング/検証セットのメトリクスを出力します。
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format。印刷形式
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))  # ログを出力
        if self.nt_per_class.sum() == 0:  # クラスごとのターゲット数の合計が0の場合
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")  # 警告ログを出力

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):  # 詳細モード、トレーニングモードではない、クラス数が1より大きい、統計がある場合
            for i, c in enumerate(self.metrics.ap_class_index):  # クラスを反復処理
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )  # ログを出力

        if self.args.plots:  # プロットがTrueの場合
            for normalize in True, False:  # 正規化を反復処理
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )  # 混同行列をプロット

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        # 正しい予測行列を返します。
        #
        # 引数：
        #     detections (torch.Tensor): 各検出が(x1、y1、x2、y2、conf、クラス)である検出を表す形状(N、6)のテンソル。
        #     gt_bboxes (torch.Tensor): グランドトゥルースのバウンディングボックス座標を表す形状(M、4)のテンソル。各バウンディングボックスの形式は(x1、y1、x2、y2)です。
        #     gt_cls (torch.Tensor): ターゲットクラスインデックスを表す形状(M,)のテンソル。
        #
        # 戻り値：
        #     (torch.Tensor): 10個のIoUレベルに対する形状(N、10)の正しい予測行列。
        #
        # 注：
        #     この関数は、メトリック計算に直接使用できる値を返しません。代わりに、グランドトゥルースに対して予測を評価するために使用される中間表現を提供します。
        iou = box_iou(gt_bboxes, detections[:, :4])  # IoUを計算
        return self.match_predictions(detections[:, 5], gt_cls, iou)  # 予測を照合

    def build_dataset(self, img_path, mode="val", batch=None):
        # YOLOデータセットを構築します。
        #
        # 引数：
        #     img_path (str): 画像を含むフォルダへのパス。
        #     mode (str): 「train」モードまたは「val」モード。ユーザーは各モードに対して異なる拡張機能をカスタマイズできます。
        #     batch (int, optional): バッチのサイズ。「rect」用です。デフォルトはNoneです。
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)  # YOLOデータセットを構築して返す

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        # データローダーを構築して返します。
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")  # データセットを構築
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # データローダーを構築して返す

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        # 検証画像サンプルをプロットします。
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 画像をプロット

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        # 入力画像に予測されたバウンディングボックスをプロットし、結果を保存します。
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        # 特定の形式で、正規化された座標でYOLO検出をtxtファイルに保存します。
        from engine.results import Results  # 結果クラスをインポート

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)  # txtファイルに保存

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        # YOLO予測をCOCO json形式にシリアル化します。
        stem = Path(filename).stem  # ファイル名を取得
        image_id = int(stem) if stem.isnumeric() else stem  # 画像IDを設定
        box = ops.xyxy2xywh(predn[:, :4])  # xywhに変換
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner。中心を左上隅に移動
        for p, b in zip(predn.tolist(), box.tolist()):  # 予測とボックスを反復処理
            self.jdict.append(  # JSONディクショナリに追加
                {
                    "image_id": image_id,  # 画像ID
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis。インデックスはlvisの場合は1から始まります
                    "bbox": [round(x, 3) for x in b],  # バウンディングボックス
                    "score": round(p[4], 5),  # スコア
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        # JSON形式でYOLO出力を評価し、パフォーマンス統計を返します。
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):  # JSONを保存、COCOまたはLVIS、jdictがある場合
            pred_json = self.save_dir / "predictions.json"  # predictions。予測
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations。アノテーション
            pkg = "pycocotools" if self.is_coco else "lvis"  # パッケージ名
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")  # ログを出力
            try:  # 例外処理
                for x in pred_json, anno_json:  # ファイルを反復処理
                    assert x.is_file(), f"{x} file not found"  # ファイルが存在するか確認
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")  # 要件をチェック
                if self.is_coco:  # COCOの場合
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api。アノテーションAPIを初期化
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)。予測APIを初期化
                    val = COCOeval(anno, pred, "bbox")  # COCO評価を初期化
                else:  # LVISの場合
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api。アノテーションAPIを初期化
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)。予測APIを初期化
                    val = LVISEval(anno, pred, "bbox")  # LVIS評価を初期化
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval。評価する画像
                val.evaluate()  # 評価
                val.accumulate()  # 蓄積
                val.summarize()  # 要約
                if self.is_lvis:  # LVISの場合
                    val.print_results()  # explicitly call print_results。明示的にprint_resultsを呼び出す
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )  # 統計を更新
            except Exception as e:  # 例外が発生した場合
                LOGGER.warning(f"{pkg} unable to run: {e}")  # 警告ログを出力
        return stats  # 統計を返す