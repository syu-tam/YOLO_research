
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""
# データセットでモデルをトレーニングします。
#
# 使用法：
#     $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16

import gc  # ガベージコレクションインターフェースを提供します。
import math  # 数学関数を提供します。
import os  # さまざまなオペレーティングシステム関数へのアクセスを提供します。
import subprocess  # サブプロセスを作成するためのモジュール
import time  # 時間関連の関数を提供します。
import warnings  # 警告メッセージの処理を可能にします。
from copy import copy, deepcopy  # オブジェクトのコピー操作を提供します。
from datetime import datetime, timedelta  # 日付と時間の操作を行うためのクラスを提供します。
from pathlib import Path  # ファイルシステムパスを操作するためのクラスを提供します。

import numpy as np  # 数値計算ライブラリ。
import torch  # PyTorchのメインモジュール。
from torch import distributed as dist  # 分散トレーニングをサポートするPyTorchモジュール。
from torch import nn, optim  # ニューラルネットワークモジュールと最適化アルゴリズムを提供します。

from cfg import get_cfg, get_save_dir  # 設定関連の関数をインポート
from data.utils import check_cls_dataset, check_det_dataset  # データセットチェック関数をインポート
from nn.tasks import attempt_load_one_weight, attempt_load_weights  # モデルのロード関数をインポート
from utils import (  # ユーティリティ関数をインポート
    DEFAULT_CFG,  # デフォルト設定
    LOCAL_RANK,  # ローカルランク
    LOGGER,  # ロガー
    RANK,  # ランク
    TQDM,  # プログレスバー
    callbacks,  # コールバック関数
    clean_url,  # URLをクリーンアップする関数
    colorstr,  # カラー文字列
    emojis,  # 絵文字
    yaml_save,  # YAML保存関数
)
from utils.autobatch import check_train_batch_size  # 自動バッチサイズチェック関数をインポート
from utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args  # チェック関数をインポート
from utils.dist import ddp_cleanup, generate_ddp_command  # 分散トレーニング関連の関数をインポート
from utils.files import get_latest_run  # 最新の実行を取得する関数をインポート
from utils.torch_utils import (  # PyTorchユーティリティ関数をインポート
    TORCH_2_4,  # PyTorchのバージョンが2.4以降かどうか
    EarlyStopping,  # 早期停止クラス
    ModelEMA,  # モデルEMAクラス
    autocast,  # 自動混合精度
    convert_optimizer_state_dict_to_fp16,  # オプティマイザの状態辞書をfp16に変換
    init_seeds,  # シードを初期化
    one_cycle,  # OneCycleLRスケジューラ
    select_device,  # デバイスを選択
    strip_optimizer,  # オプティマイザを削除
    torch_distributed_zero_first,  # 分散トレーニング用のユーティリティ
    unset_deterministic,
    )


class BaseTrainer:
    # トレーナーを作成するための基本クラス。
    #
    # 属性：
    #     args (SimpleNamespace): トレーナーの構成。
    #     validator (BaseValidator): バリデーターインスタンス。
    #     model (nn.Module): モデルインスタンス。
    #     callbacks (defaultdict): コールバックの辞書。
    #     save_dir (Path): 結果を保存するディレクトリ。
    #     wdir (Path): 重みを保存するディレクトリ。
    #     last (Path): 最後のチェックポイントへのパス。
    #     best (Path): 最高のチェックポイントへのパス。
    #     save_period (int): xエポックごとにチェックポイントを保存します（1未満の場合は無効）。
    #     batch_size (int): トレーニングのバッチサイズ。
    #     epochs (int): トレーニングするエポック数。
    #     start_epoch (int): トレーニングの開始エポック。
    #     device (torch.device): トレーニングに使用するデバイス。
    #     amp (bool): AMP（自動混合精度）を有効にするフラグ。
    #     scaler (amp.GradScaler): AMPの勾配スケーラー。
    #     data (str): データへのパス。
    #     trainset (torch.utils.data.Dataset): トレーニングデータセット。
    #     testset (torch.utils.data.Dataset): テストデータセット。
    #     ema (nn.Module): モデルのEMA（指数移動平均）。
    #     resume (bool): チェックポイントからトレーニングを再開します。
    #     lf (nn.Module): 損失関数。
    #     scheduler (torch.optim.lr_scheduler._LRScheduler): 学習率スケジューラ。
    #     best_fitness (float): 達成された最高の適合度。
    #     fitness (float): 現在の適合度。
    #     loss (float): 現在の損失値。
    #     tloss (float): 合計損失値。
    #     loss_names (list): 損失名のリスト。
    #     csv (Path): 結果CSVファイルへのパス。

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # BaseTrainerクラスを初期化します。
        #
        # 引数：
        #     cfg (str, optional): 構成ファイルへのパス。デフォルトはDEFAULT_CFGです。
        #     overrides (dict, optional): 構成のオーバーライド。デフォルトはNoneです。
        self.args = get_cfg(cfg, overrides)  # 設定を取得
        self.check_resume(overrides)  # レジュームをチェック
        self.device = select_device(self.args.device, self.args.batch)  # デバイスを選択
        self.validator = None  # バリデーター
        self.metrics = None  # メトリクス
        self.plots = {}  # プロット
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)  # シードを初期化

        # Dirs
        self.save_dir = get_save_dir(self.args)  # 保存ディレクトリを取得
        self.args.name = self.save_dir.name  # update name for loggers。ロガーの名前を更新
        self.wdir = self.save_dir / "weights"  # weights dir。重みディレクトリ
        if RANK in {-1, 0}:  # ランクが-1または0の場合
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir。ディレクトリを作成
            self.args.save_dir = str(self.save_dir)  # モデルの保存先
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args。実行引数を保存
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths。チェックポイントパス
        self.save_period = self.args.save_period  # 保存間隔

        self.batch_size = self.args.batch  # バッチサイズ
        self.epochs = self.args.epochs  # エポック数
        self.start_epoch = 0  # 開始エポック
        if RANK == -1:  # ランクが-1の場合
            print_args(vars(self.args))  # 引数を出力

        # Device
        if self.device.type in {"cpu", "mps"}:  # デバイスタイプがcpuまたはmpsの場合
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading。推論によって時間が支配されるため、CPUトレーニングが高速化

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt。サフィックスを追加
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times。データセットの自動ダウンロードの繰り返しを回避
            self.trainset, self.testset = self.get_dataset()  # データセットを取得
        self.ema = None  # EMAを初期化

        # Optimization utils init
        self.lf = None  # 損失関数を初期化
        self.scheduler = None  # スケジューラを初期化

        # Epoch level metrics
        self.best_fitness = None  # 最高の適合度を初期化
        self.fitness = None  # 適合度を初期化
        self.loss = None  # 損失を初期化
        self.tloss = None  # 合計損失を初期化
        self.loss_names = ["Loss"]  # 損失名を設定
        self.csv = self.save_dir / "results.csv"  # csvパスを設定
        self.plot_idx = [0, 1, 2]  # プロットインデックス

        # HUB
        self.hub_session = None  # HUBセッションを初期化

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # コールバックを取得
        if RANK in {-1, 0}:  # ランクが-1または0の場合
            callbacks.add_integration_callbacks(self)  # 統合コールバックを追加

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        # 指定されたコールバックを追加します。
        self.callbacks[event].append(callback)  # コールバックを追加

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        # 指定されたコールバックで既存のコールバックをオーバーライドします。
        self.callbacks[event] = [callback]  # コールバックを設定

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        # 特定のイベントに関連付けられた既存のコールバックをすべて実行します。
        for callback in self.callbacks.get(event, []):  # イベントに関連付けられたコールバックを反復処理
            callback(self)  # コールバックを実行

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # 複数GPUシステムでdevice = ''、device = Noneを使用できるように、デフォルトでdevice = 0にします。
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'。deviceが文字列で、長さがある場合
            world_size = len(self.args.device.split(","))  # デバイス数を計算
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)。deviceがタプルまたはリストの場合
            world_size = len(self.args.device)  # デバイス数を計算
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'。deviceがcpuまたはmpsの場合
            world_size = 0  # ワールドサイズを0に設定
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number。cudaが利用可能な場合
            world_size = 1  # default to device 0。デフォルトでdevice 0
        else:  # i.e. device=None or device=''。それ以外の場合
            world_size = 0  # ワールドサイズを0に設定

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:  # ワールドサイズが1より大きく、LOCAL_RANKが環境変数にない場合
            # Argument checks
            if self.args.rect:  # アスペクト比固定の場合
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")  # 警告ログを出力
                self.args.rect = False  # アスペクト比固定を無効化
            if self.args.batch < 1.0:  # バッチサイズが1より小さい場合
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )  # 警告ログを出力
                self.args.batch = 16  # バッチサイズを16に設定

            # Command
            cmd, file = generate_ddp_command(world_size, self)  # DDPコマンドを生成
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')  # DDPコマンドを出力
                subprocess.run(cmd, check=True)  # DDPコマンドを実行
            except Exception as e:  # エラーが発生した場合
                raise e  # エラーを発生
            finally:  # 最後に
                ddp_cleanup(self, str(file))  # DDPをクリーンアップ

        else:  # 通常トレーニング
            self._do_train(world_size)  # トレーニングを実行

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        # トレーニング学習率スケジューラを初期化します。
        if self.args.cos_lr:  # コサイン学習率を使用する場合
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']。コサインスケジューラを初期化
        else:  # 線形学習率を使用する場合
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear。線形スケジューラを初期化
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)  # ラムダLRスケジューラを初期化

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        # トレーニング用にDistributedDataParallelパラメータを初期化および設定します。
        torch.cuda.set_device(RANK)  # デバイスを設定
        self.device = torch.device("cuda", RANK)  # デバイスを設定
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout。タイムアウトを強制するために設定
        dist.init_process_group(  # プロセスグループを初期化
            backend="nccl" if dist.is_nccl_available() else "gloo",  # バックエンドを設定
            timeout=timedelta(seconds=10800),  # 3 hours。タイムアウトを設定
            rank=RANK,  # ランクを設定
            world_size=world_size,  # ワールドサイズを設定
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        # 正しいランクプロセスでデータローダーとオプティマイザを構築します。
        # Model
        self.run_callbacks("on_pretrain_routine_start")  # pretrainルーチンの開始時にコールバックを実行
        ckpt = self.setup_model()  # モデルをセットアップ
        self.model = self.model.to(self.device)  # モデルをデバイスに移動
        self.set_model_attributes()  # モデル属性を設定

        # Freeze layers
        freeze_list = (  # フリーズするレイヤーのリストを取得
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers。常にこれらのレイヤーをフリーズ
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names  # フリーズするレイヤー名
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():  # モデルのパラメータを反復処理
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):  # レイヤー名がフリーズリストに含まれている場合
                LOGGER.info(f"Freezing layer '{k}'")  # ログを出力
                v.requires_grad = False  # 勾配計算を無効化
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients。浮動小数点テンソルのみが勾配を必要とする
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )  # 警告ログを出力
                v.requires_grad = True  # 勾配計算を有効化

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False。AMPを有効にするかどうか
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP。シングルGPUとDDPの場合
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them。check_amp（）がリセットするため、コールバックをバックアップ
            self.amp = torch.tensor(check_amp(self.model), device=self.device)  # check amp。AMPをチェック
            callbacks.default_callbacks = callbacks_backup  # restore callbacks。コールバックを復元
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)。ランク0から他のすべてのランクにテンソルをブロードキャスト
        self.amp = bool(self.amp)  # as boolean。ブール値として
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )  # 勾配スケーラーを初期化
        if world_size > 1:  # ワールドサイズが1より大きい場合
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)  # DDPを初期化

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)。グリッドサイズ
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)  # 画像サイズをチェック
        self.stride = gs  # for multiscale training。マルチスケールトレーニング用

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size。シングルGPUの場合のみ、最適なバッチサイズを見積もる
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)  # バッチサイズを計算
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")  # トレーニングデータローダーを取得
        if RANK in {-1, 0}:  # ランクが-1または0の場合
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )  # テストデータローダーを取得
            self.validator = self.get_validator()  # バリデーターを取得
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")  # メトリックキーを取得
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # メトリクスを初期化
            self.ema = ModelEMA(self.model)  # EMAを初期化
            if self.args.plots:  # プロットする場合
                self.plot_training_labels()  # トレーニングラベルをプロット

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing。最適化する前に損失を累積
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay。重み減衰をスケール
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs  # 反復回数を計算
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )  # オプティマイザを構築
        # Scheduler
        self._setup_scheduler()  # スケジューラをセットアップ
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False  # 早期停止を設定
        self.resume_training(ckpt)  # トレーニングを再開
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move。移動しないでください
        self.run_callbacks("on_pretrain_routine_end")  # pretrainルーチンの終了時にコールバックを実行

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        # 引数で指定された場合は、トレーニングを完了し、評価およびプロットします。
        if world_size > 1:  # ワールドサイズが1より大きい場合
            self._setup_ddp(world_size)  # DDPをセットアップ
        self._setup_train(world_size)  # トレーニングをセットアップ

        nb = len(self.train_loader)  # number of batches。バッチ数
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations。ウォームアップ反復
        last_opt_step = -1  # 最後の最適化ステップ
        self.epoch_time = None  # エポック時間
        self.epoch_time_start = time.time()  # エポック開始時間
        self.train_time_start = time.time()  # トレーニング開始時間
        self.run_callbacks("on_train_start")  # トレーニング開始時にコールバックを実行
        LOGGER.info(  # ログを出力
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:  # モザイクを閉じる場合
            base_idx = (self.epochs - self.args.close_mosaic) * nb  # 基本インデックス
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])  # プロットインデックスを拡張
        epoch = self.start_epoch  # エポックを開始エポックに設定
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start。トレーニング開始時の安定性を確保するために、再開された勾配をゼロにする
        while True:  # 無限ループ
            self.model.epoch = epoch  # プロパティを通して_epochを更新
            self.model.total_epochs = self.epochs  # プロパティを通して_total_epochsを更新
            self.epoch = epoch  # エポックを設定
            self.run_callbacks("on_train_epoch_start")
            
            with warnings.catch_warnings():  # 警告をキャッチ
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'。'optimizer.step（）の前にlr_scheduler.step（）が検出されました'を抑制
                self.scheduler.step()  # スケジューラーステップ

            self.model.train()  # モデルをトレーニングモードに設定
            if RANK != -1:  # DDPトレーニングの場合
                self.train_loader.sampler.set_epoch(epoch)  # サンプラーのエポックを設定
            pbar = enumerate(self.train_loader)  # データローダーを列挙
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):  # モザイクを閉じる場合
                self._close_dataloader_mosaic()  # データローダーモザイクを閉じる
                self.train_loader.reset()  # トレーニングデータローダーをリセット

            if RANK in {-1, 0}:  # ランクが-1または0の場合
                LOGGER.info(self.progress_string())  # 進行状況文字列を記録
                pbar = TQDM(enumerate(self.train_loader), total=nb)  # TQDMプログレスバーを初期化
            self.tloss = None  # 合計損失をリセット
            for i, batch in pbar:  # バッチを反復処理
                self.run_callbacks("on_train_batch_start")  # トレーニングバッチの開始時にコールバックを実行
                # Warmup
                ni = i + nb * epoch  # 反復回数を計算
                if ni <= nw:  # ウォームアップの場合
                    xi = [0, nw]  # x interp。x補間
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))  # 蓄積数を計算
                    for j, x in enumerate(self.optimizer.param_groups):  # オプティマイザパラメータグループを反復処理
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )  # 学習率を計算
                        if "momentum" in x:  # モメンタムがある場合
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])  # モメンタムを計算

                # Forward
                with autocast(self.amp):  # 自動混合精度を使用
                    batch = self.preprocess_batch(batch)  # バッチを前処理
                    loss, self.loss_items = self.model(batch)  # モデルを適用
                    self.loss = loss.sum()
                    if RANK != -1:  # DDPトレーニングの場合
                        self.loss *= world_size  # 損失をスケール
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )  # 合計損失を更新

                # Backward
                self.scaler.scale(self.loss).backward()  # 損失をスケールして逆伝播

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:  # 蓄積数が蓄積数以上の場合
                    self.optimizer_step()  # オプティマイザーステップ
                    last_opt_step = ni  # 最後の最適化ステップを更新

                    # Timed stopping
                    if self.args.time:  # 時間制限がある場合
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)  # トレーニング時間を超えたかどうか
                        if RANK != -1:  # if DDP training。DDPトレーニングの場合
                            broadcast_list = [self.stop if RANK == 0 else None]  # ブロードキャストリスト
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks。すべてのランクに「停止」をブロードキャスト
                            self.stop = broadcast_list[0]  # 停止状態を更新
                        if self.stop:  # training time exceeded。トレーニング時間が超過した場合
                            break  # トレーニングループをブレイク

                # Log
                if RANK in {-1, 0}:  # ランクが-1または0の場合
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1  # 損失の長さを計算
                    pbar.set_description(  # プログレスバーの説明を設定
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",  # エポック
                            f"{self._get_memory():.3g}G",  # （GB）GPUメモリ使用率
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 損失
                            batch["cls"].shape[0],  # バッチサイズ、つまり8
                            batch["img"].shape[-1],  # imgsz、つまり640
                        )
                    )
                    self.run_callbacks("on_batch_end")  # バッチ終了時にコールバックを実行
                    if self.args.plots and ni in self.plot_idx:  # プロットする場合
                        self.plot_training_samples(batch, ni)  # トレーニングサンプルをプロット

                self.run_callbacks("on_train_batch_end")  # トレーニングバッチの終了時にコールバックを実行

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers。ロガー用
            self.run_callbacks("on_train_epoch_end")  # トレーニングエポックの終了時にコールバックを実行
            if RANK in {-1, 0}:  # ランクが-1または0の場合
                final_epoch = epoch + 1 >= self.epochs  # 最後のepochかどうか
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])  # EMAの属性を更新


                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                     self.metrics, self.fitness = self.validate()  # バリデーションを実行
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})  # メトリクスを保存
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch  # 早期停止を適用
                if self.args.time:  # 時間制限がある場合
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)  # 時間を超過したかどうか

                # Save model
                if self.args.save or final_epoch:  # 保存または最終エポックの場合
                    self.save_model()  # モデルを保存
                    self.run_callbacks("on_model_save")  # モデル保存時にコールバックを実行

            # Scheduler
            t = time.time()  # 現在時刻を取得
            self.epoch_time = t - self.epoch_time_start  # エポック時間を計算
            self.epoch_time_start = t  # エポック開始時間を更新
            if self.args.time:  # 時間制限がある場合
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)  # 平均エポック時間を計算
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)  # エポック数を再計算
                self._setup_scheduler()  # スケジューラを再セットアップ
                self.scheduler.last_epoch = self.epoch  # do not move。移動しないでください
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs。エポックを超過した場合に停止
            self.run_callbacks("on_fit_epoch_end")  # フィットエポック終了時にコールバックを実行
            if self._get_memory(fraction=True) > 0.5:
                self._clear_memory()  # メモリをクリア

            # Early Stopping
            if RANK != -1:  # if DDP training。DDPトレーニングの場合
                broadcast_list = [self.stop if RANK == 0 else None]  # ブロードキャストリスト
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks。すべてのランクに「停止」をブロードキャスト
                self.stop = broadcast_list[0]  # 停止状態を更新
            if self.stop:  # 停止する場合
                break  # must break all DDP ranks。すべてのDDPランクを中断する必要がある
            epoch += 1  # エポックをインクリメント

        if RANK in {-1, 0}:  # ランクが-1または0の場合
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start  # トレーニング時間を計算
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")  # ログを出力
            self.final_eval()  # 最終評価を実行
            if self.args.plots:  # プロットする場合
                self.plot_metrics()  # メトリクスをプロット
            self.run_callbacks("on_train_end")  # トレーニング終了時にコールバックを実行
        self._clear_memory()  # メモリをクリア
        unset_deterministic()
        self.run_callbacks("teardown")  # ティアダウン時にコールバックを実行

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size
    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB."""
        # GB単位のアクセラレータメモリ使用率を取得します。
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        # 異なるプラットフォームでアクセラレータメモリをクリアします。
        gc.collect()  # ガベージコレクションを実行
        if self.device.type == "mps":  # デバイスタイプがmpsの場合
            torch.mps.empty_cache()  # MPSキャッシュをクリア
        elif self.device.type == "cpu":  # デバイスタイプがcpuの場合
            return  # 何もしない
        else:  # それ以外の場合
            torch.cuda.empty_cache()  # CUDAキャッシュをクリア

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        # pandasを使用してresults.csvをdictに読み込みます。
        import pandas as pd  # scope for faster 'import ultralytics'。より高速な「import ultralytics」のスコープ

        return pd.read_csv(self.csv).to_dict(orient="list")  # csvを読み取って辞書に変換
    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()
    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        # 追加のメタデータを使用してモデルトレーニングチェックポイントを保存します。
        import io  # ioモジュールをインポート

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()  # バイトIOバッファを作成
        torch.save(  # チェックポイントをシリアル化
            {
                "epoch": self.epoch,  # エポック
                "best_fitness": self.best_fitness,  # 最高の適合度
                "model": None,  # resume and final checkpoints derive from EMA。レジュームと最終チェックポイントはEMAから派生
                "ema": deepcopy(self.ema.ema).half(),  # EMA
                "updates": self.ema.updates,  # アップデート
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),  # オプティマイザ
                "train_args": vars(self.args),  # save as dict。辞書として保存
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},  # train metrics。トレーニングメトリクス
                "train_results": self.read_results_csv(),  # トレーニング結果
                "date": datetime.now().isoformat(),  # 現在日時
                "license": "AGPL-3.0 (https://ultralytics.com/license)",  # ライセンス
                "docs": "https://docs.ultralytics.com",  # ドキュメント
            },
            buffer,  # バッファに保存
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save。保存するシリアル化されたコンテンツを取得

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt。last.ptを保存
        if self.best_fitness == self.fitness:  # 最高の適合度の場合
            self.best.write_bytes(serialized_ckpt)  # save best.pt。best.ptを保存
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):  # 保存期間の場合
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'。epochを保存

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        # 存在する場合、データ辞書からtrain、valパスを取得します。
        # データ形式が認識されない場合はNoneを返します。
        try:
            if self.args.task == "classify":  # 分類タスクの場合
                data = check_cls_dataset(self.args.data)  # 分類データセットをチェック
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:  # データ形式がyamlまたはymlの場合
                data = check_det_dataset(self.args.data)  # 検出データセットをチェック
                if "yaml_file" in data:  # yaml_fileがある場合
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage。 'yolo train data=url.zip'の使用法を検証するため
        except Exception as e:  # 例外が発生した場合
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e  # エラーメッセージを生成
        self.data = data  # データセットをアタッチ
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "item"}
            self.data["nc"] = 1
        return data["train"], data.get("val") or data.get("test")  # トレーニングパスと検証パスを返す

    def setup_model(self):
        """Load/create/download model for any task."""
        # タスクのモデルをロード/作成/ダウンロードします。
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed。モデルが事前にロードされている場合。セットアップは不要
            return  # 何もしない

        cfg, weights = self.model, None  # 設定と重みを初期化
        ckpt = None  # チェックポイントを初期化
        if str(self.model).endswith(".pt"):  # モデルが.ptで終わる場合
            weights, ckpt = attempt_load_one_weight(self.model)  # モデルとチェックポイントをロード
            cfg = weights.yaml  # 設定をロード
        elif isinstance(self.args.pretrained, (str, Path)):  # 事前トレーニング済みが指定されている場合
            weights, _ = attempt_load_one_weight(self.args.pretrained)  # 事前トレーニング済みモデルをロード
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)。Model（cfg、weights）を呼び出す
        return ckpt  # チェックポイントを返す

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # 勾配クリッピングとEMA更新を使用して、トレーニングオプティマイザーの単一のステップを実行します。
        self.scaler.unscale_(self.optimizer)  # unscale gradients。勾配をアン スケール
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients。勾配をクリップ
        self.scaler.step(self.optimizer)  # オプティマイザステップ
        self.scaler.update()  # スケーラーを更新
        self.optimizer.zero_grad()  # 勾配をゼロ
        if self.ema:  # EMAがある場合
            self.ema.update(self.model)  # EMAを更新

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        # タスクタイプに応じて、カスタム前処理モデル入力とグランドトゥルースを許可します。
        return batch  # バッチを返す

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        # self.validatorを使用してテストセットで検証を実行します。
        # 返されるdictは、「fitness」キーを含むことが期待されます。
        metrics = self.validator(self)  # バリデーションを実行
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found。適合度が見つからない場合は、損失を適合度の尺度として使用
        if not self.best_fitness or self.best_fitness < fitness: # 最高の適合度がない、または最高の適合度より適合度が高い場合
            print('\033[34m' + f"best fitness updated to {self.epoch + 1}" + '\033[0m') # ログを出力
            self.best_fitness = fitness  # 最高の適合度を更新
        return metrics, fitness  # メトリクスと適合度を返す

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        # モデルを取得し、cfgファイルのロードに対してNotImplementedErrorを発生させます。
        raise NotImplementedError("This task trainer doesn't support loading cfg files")  # エラーを発生

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        # get_validator関数が呼び出されたときにNotImplementedErrorを返します。
        raise NotImplementedError("get_validator function not implemented in trainer")  # エラーを発生

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        # torch.data.Dataloaderから派生したdataloaderを返します。
        raise NotImplementedError("get_dataloader function not implemented in trainer")  # エラーを発生

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        # データセットを構築します。
        raise NotImplementedError("build_dataset function not implemented in trainer")  # エラーを発生

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        # ラベル付けされたトレーニング損失アイテムテンソルを含む損失dictを返します。
        #
        # 注：
        #     これは分類には必要ありませんが、セグメンテーションと検出には必要です
        return {"loss": loss_items} if loss_items is not None else ["loss"]  # 損失アイテムがある場合は損失ディクショナリを返し、それ以外の場合は「loss」を返します

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        # トレーニング前にモデルパラメータを設定または更新します。
        self.model.names = self.data["names"]  # クラス名を設定

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        # YOLOモデルをトレーニングするためのターゲットテンソルを構築します。
        pass  # 何もしない

    def progress_string(self):
        """Returns a string describing training progress."""
        # トレーニングの進行状況を説明する文字列を返します。
        return ""  # 空の文字列を返す

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        # YOLOトレーニング中にトレーニングサンプルをプロットします。
        pass  # 何もしない

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        # YOLOモデルのトレーニングラベルをプロットします。
        pass  # 何もしない

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        # トレーニングメトリックをCSVファイルに保存します。
        keys, vals = list(metrics.keys()), list(metrics.values())  # メトリクスのキーと値を取得
        n = len(metrics) + 2  # number of cols。列数
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header。ヘッダー
        t = time.time() - self.train_time_start  # 時間を計算
        with open(self.csv, "a", encoding="utf-8") as f:  # ファイルを開く
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")  # メトリクスを保存

    def plot_metrics(self):
        """Plot and display metrics visually."""
        # メトリクスをプロットして視覚的に表示します。
        pass  # 何もしない

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        # プロットを登録します（例：コールバックで使用するため）。
        path = Path(name)  # パスを生成
        self.plots[path] = {"data": data, "timestamp": time.time()}  # プロットを登録

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        # オブジェクト検出YOLOモデルの最終評価と検証を実行します。
        ckpt = {}  # チェックポイントを初期化
        for f in self.last, self.best:  # 最後のモデルと最高のモデルを反復処理
            if f.exists():  # ファイルが存在する場合
                if f is self.last:  # 最後のモデルの場合
                    ckpt = strip_optimizer(f)  # オプティマイザを削除
                elif f is self.best:  # 最高のモデルの場合
                    k = "train_results"  # update best.pt train_metrics from last.pt。last.ptからbest.pt train_metricsを更新
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)  # オプティマイザを削除
                    LOGGER.info(f"\nValidating {f}...")  # ログを出力
                    self.validator.args.plots = self.args.plots  # プロット引数を設定
                    self.metrics = self.validator(model=f)  # バリデーションを実行
                    self.metrics.pop("fitness", None)  # 適合度を削除
                    self.run_callbacks("on_fit_epoch_end")  # フィットエポック終了時にコールバックを実行

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        # レジュームチェックポイントが存在するかどうかを確認し、それに応じて引数を更新します。
        resume = self.args.resume  # レジューム引数を取得
        if resume:  # レジュームがTrueの場合
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()  # レジュームが存在するかどうかをチェック
                last = Path(check_file(resume) if exists else get_latest_run())  # 最後の実行を取得

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args  # レジュームチェックポイント引数をロード
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data  # データYAMLを更新

                resume = True  # レジュームをTrueに設定
                self.args = get_cfg(ckpt_args)  # 設定を取得
                self.args.model = self.args.resume = str(last)  # 最後のモデルで再設定
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume。メモリを削減したり、再開時にデバイスを更新したりするための引数更新を許可
                    if k in overrides:  # 引数がオーバーライドに含まれている場合
                        setattr(self.args, k, overrides[k])  # 引数を設定

            except Exception as e:  # 例外が発生した場合
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e  # 例外を発生
        self.resume = resume  # レジュームを設定

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        # 指定されたエポックと最高の適合度からYOLOトレーニングを再開します。
        if ckpt is None or not self.resume:  # チェックポイントがない場合、または再開しない場合
            return  # 何もしない
        best_fitness = 0.0  # 最高の適合度を初期化
        start_epoch = ckpt.get("epoch", -1) + 1  # 開始エポックを取得
        if ckpt.get("optimizer", None) is not None:  # オプティマイザがある場合
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer。オプティマイザをロード
            best_fitness = ckpt["best_fitness"]  # 最高の適合度を取得
        if self.ema and ckpt.get("ema"):  # EMAがあり、チェックポイントにEMAがある場合
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMAをロード
            self.ema.updates = ckpt["updates"]  # アップデートを取得
        assert start_epoch > 0, (  # 開始エポックが0より大きいことを確認
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")  # ログを出力
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness  # 最高の適合度を設定
        self.start_epoch = start_epoch  # 開始エポックを設定
        if start_epoch > (self.epochs - self.args.close_mosaic):  # 開始エポックがclose_mosaicを超える場合
            self._close_dataloader_mosaic()  # データローダーモザイクを閉じる


    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        # データローダーを更新して、モザイク拡張の使用を停止します。
        if hasattr(self.train_loader.dataset, "mosaic"):  # トレーニングデータローダーにモザイクがある場合
            self.train_loader.dataset.mosaic = False  # モザイクを無効化
        if hasattr(self.train_loader.dataset, "close_mosaic"):  # トレーニングデータローダーにclose_mosaicがある場合
            LOGGER.info("Closing dataloader mosaic")  # ログを出力
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))  # モザイクを閉じる

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        # 指定されたオプティマイザ名、学習率、モメンタム、重み減衰、および反復回数に基づいて、指定されたモデルのオプティマイザを構築します。
        #
        # 引数：
        #     model (torch.nn.Module): オプティマイザを構築するモデル。
        #     name (str, optional): 使用するオプティマイザの名前。 'auto'の場合、オプティマイザは反復回数に基づいて選択されます。
        #         デフォルト：「auto」。
        #     lr (float, optional): オプティマイザの学習率。デフォルト：0.001。
        #     momentum (float, optional): オプティマイザのモメンタムファクター。デフォルト：0.9。
        #     decay (float, optional): オプティマイザの重み減衰。デフォルト：1e-5。
        #     iterations (float, optional): 反復回数。nameが 'auto'の場合にオプティマイザを決定します。デフォルト：1e5。
        #
        # 戻り値：
        #     (torch.optim.Optimizer): 構築されたオプティマイザ。
        g = [], [], []  # optimizer parameter groups。オプティマイザパラメータグループ
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()。正規化レイヤー
        if name == "auto":  # オプティマイザ名がautoの場合
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )  # ログを出力
            nc = self.data.get("nc", 10)  # number of classes。クラス数
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places。6桁の10進数へのlr0適合方程式
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)  # オプティマイザ、学習率、モメンタムを設定
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam。Adamの場合は0.01を超えない

        for module_name, module in model.named_modules():  # モデルモジュールを反復処理
            for param_name, param in module.named_parameters(recurse=False):  # モデルパラメータを反復処理
                fullname = f"{module_name}.{param_name}" if module_name else param_name  # 完全名を取得
                if "bias" in fullname:  # bias (no decay)。バイアス（減衰なし）
                    g[2].append(param)  # バイアスパラメータをリストに追加
                elif isinstance(module, bn):  # weight (no decay)。重み（減衰なし）
                    g[1].append(param)  # 正規化レイヤをリストに追加
                else:  # weight (with decay)。重み（減衰あり）
                    g[0].append(param)  # その他のパラメータをリストに追加
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:  # アダム系オプティマイザの場合
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)  # オプティマイザを初期化
        elif name == "RMSProp":  # RMSPropオプティマイザの場合
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)  # オプティマイザを初期化
        elif name == "SGD":  # SGDオプティマイザの場合
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)  # オプティマイザを初期化
        else:  # サポートされていないオプティマイザの場合
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )  # エラーを発生

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay。重み減衰でg0を追加
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)。BatchNorm2d重みを追加
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )  # ログを出力
        return optimizer  # オプティマイザを返す