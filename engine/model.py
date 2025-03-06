# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect  # モジュール内のオブジェクトに関する情報を取得するためのモジュール
from pathlib import Path  # ファイルやディレクトリをオブジェクトとして扱うためのモジュール
from typing import List, Union  # 型ヒントで使用するためのモジュール

import numpy as np  # 数値計算ライブラリ
import torch  # PyTorch深層学習フレームワーク
from PIL import Image  # 画像処理ライブラリ

from cfg import TASK2DATA, get_cfg, get_save_dir  # 設定関連の関数をインポート
from .results import Results  # モデルの出力結果を格納するクラスをインポート
from hub import HUB_WEB_ROOT, HUBTrainingSession  # Ultralytics HUBとの連携に関するクラスをインポート
from nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load  # モデルのロードやタスクの推測に関する関数をインポート
from utils import (  # ユーティリティ関数をインポート
    ARGV,  # コマンドライン引数
    ASSETS,  # アセットファイルへのパス
    DEFAULT_CFG_DICT,  # デフォルトの設定辞書
    LOGGER,  # ロガー
    RANK,  # プロセスランク（分散学習用）
    SETTINGS,  # 設定
    callbacks,  # コールバック関数
    checks,  # チェック関数
    emojis,  # 絵文字
    yaml_load,  # YAMLファイルのロード関数
)


class Model(nn.Module):  # PyTorchのnn.Moduleを継承したModelクラス
    # YOLOモデルを実装するための基本クラス。異なるモデルタイプ間でAPIを統一します。
    # このクラスは、トレーニング、検証、予測、エクスポート、ベンチマークなど、YOLOモデルに関連するさまざまな操作のための共通インターフェースを提供します。
    # ローカルファイル、Ultralytics HUB、またはTriton Serverからロードされたものを含む、さまざまなタイプのモデルを処理します。
    #
    # 属性:
    #     callbacks (Dict): モデル操作中のさまざまなイベントに対するコールバック関数の辞書。
    #     predictor (BasePredictor): 予測を行うために使用されるpredictorオブジェクト。
    #     model (nn.Module): 基盤となるPyTorchモデル。
    #     trainer (BaseTrainer): モデルのトレーニングに使用されるtrainerオブジェクト。
    #     ckpt (Dict): モデルが*.ptファイルからロードされた場合のチェックポイントデータ。
    #     cfg (str): モデルが*.yamlファイルからロードされた場合の設定。
    #     ckpt_path (str): チェックポイントファイルへのパス。
    #     overrides (Dict): モデル設定のオーバーライドの辞書。
    #     metrics (Dict): 最新のトレーニング/検証メトリクス。
    #     session (HUBTrainingSession): Ultralytics HUBセッション（該当する場合）。
    #     task (str): モデルが対象とするタスクのタイプ。
    #     model_name (str): モデルの名前。
    #
    # メソッド:
    #     __call__: predictメソッドのエイリアスで、モデルインスタンスを呼び出し可能にします。
    #     _new: 設定ファイルに基づいて新しいモデルを初期化します。
    #     _load: チェックポイントファイルからモデルをロードします。
    #     _check_is_pytorch_model: モデルがPyTorchモデルであることを確認します。
    #     reset_weights: モデルの重みを初期状態にリセットします。
    #     load: 指定されたファイルからモデルの重みをロードします。
    #     save: モデルの現在の状態をファイルに保存します。
    #     info: モデルに関する情報をログまたは返します。
    #     fuse: 最適化された推論のためにConv2dレイヤーとBatchNorm2dレイヤーを融合します。
    #     predict: オブジェクト検出予測を実行します。
    #     track: オブジェクト追跡を実行します。
    #     val: データセットでモデルを検証します。
    #     benchmark: さまざまなエクスポート形式でモデルをベンチマークします。
    #     export: モデルをさまざまな形式にエクスポートします。
    #     train: データセットでモデルをトレーニングします。
    #     tune: ハイパーパラメータのチューニングを実行します。
    #     _apply: モデルのテンソルに関数を適用します。
    #     add_callback: イベントのコールバック関数を追加します。
    #     clear_callback: イベントのすべてのコールバックをクリアします。
    #     reset_callbacks: すべてのコールバックをデフォルト関数にリセットします。
    #
    # 例:
    #     >>> from ultralytics import YOLO
    #     >>> model = YOLO("yolo11n.pt")
    #     >>> results = model.predict("image.jpg")
    #     >>> model.train(data="coco8.yaml", epochs=3)
    #     >>> metrics = model.val()
    #     >>> model.export(format="onnx")

    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",  # モデルのパスまたは名前。デフォルトは"yolo11n.pt"
        task: str = None,  # タスクの種類。Noneの場合は自動的に推測される
        verbose: bool = False,  # 詳細なログを出力するかどうか。デフォルトはFalse
    ) -> None:
        # YOLOモデルクラスの新しいインスタンスを初期化します。
        # このコンストラクタは、提供されたモデルパスまたは名前に基づいてモデルを設定します。
        # ローカルファイル、Ultralytics HUBモデル、Triton Serverモデルなど、さまざまなタイプのモデルソースを処理します。
        # このメソッドは、モデルのいくつかの重要な属性を初期化し、トレーニング、予測、またはエクスポートなどの操作の準備をします。
        #
        # 引数:
        #     model (Union[str, Path]): ロードまたは作成するモデルのパスまたは名前。
        #         ローカルファイルパス、Ultralytics HUBからのモデル名、またはTriton Serverモデルを指定できます。
        #     task (str | None): YOLOモデルに関連付けられたタスクタイプ。そのアプリケーションドメインを指定します。
        #     verbose (bool): Trueの場合、モデルの初期化および後続の操作中に詳細な出力を有効にします。
        #
        # 例外:
        #     FileNotFoundError: 指定されたモデルファイルが存在しないか、アクセスできない場合。
        #     ValueError: モデルファイルまたは設定が無効またはサポートされていない場合。
        #     ImportError: 特定のモデルタイプ（HUB SDKなど）に必要な依存関係がインストールされていない場合。
        #
        # 例:
        #     >>> model = Model("yolo11n.pt")
        #     >>> model = Model("path/to/model.yaml", task="detect")
        #     >>> model = Model("hub_model", verbose=True)
        super().__init__()  # 親クラスのコンストラクタを呼び出す
        self.callbacks = callbacks.get_default_callbacks()  # デフォルトのコールバック関数を取得
        self.predictor = None  # 再利用可能なpredictor
        self.model = None  # モデルオブジェクト
        self.trainer = None  # trainerオブジェクト
        self.ckpt = None  # *.ptからロードした場合のチェックポイント
        self.cfg = None  # *.yamlからロードした場合の設定
        self.ckpt_path = None  # チェックポイントファイルのパス
        self.overrides = {}  # trainerオブジェクトの上書き
        self.metrics = None  # 検証/トレーニングメトリクス
        self.session = None  # HUBセッション
        self.task = task  # タスクタイプ
        model = str(model).strip()  # モデルのパスを文字列に変換し、前後の空白を削除

        # Load or create new YOLO model
        if Path(model).suffix in {".yaml", ".yml"}:  # モデルのパスがYAMLファイルの場合
            self._new(model, task=task, verbose=verbose)  # 新しいモデルを初期化
        else:  # モデルのパスがYAMLファイルではない場合
            self._load(model, task=task)  # モデルをロード

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,  # 入力ソース。ファイルパス、URL、PILイメージ、numpy配列など
        stream: bool = False,  # ストリーム入力を処理するかどうか。デフォルトはFalse
        **kwargs,  # その他のキーワード引数
    ) -> list:
        # predictメソッドのエイリアス。モデルインスタンスを呼び出し可能にして予測を実行します。
        # このメソッドは、必要な引数でモデルインスタンスを直接呼び出すことで、予測プロセスを簡素化します。
        #
        # 引数:
        #     source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): 予測を行う画像のソース。
        #         ファイルパス、URL、PILイメージ、numpy配列、PyTorchテンソル、またはこれらのリスト/タプルを指定できます。
        #     stream (bool): Trueの場合、入力ソースを予測用の連続ストリームとして扱います。
        #     **kwargs (Any): 予測プロセスを構成するための追加のキーワード引数。
        #
        # 戻り値:
        #     (List[ultralytics.engine.results.Results]): 予測結果のリスト。各結果はResultsオブジェクトにカプセル化されます。
        #
        # 例:
        #     >>> model = YOLO("yolo11n.pt")
        #     >>> results = model("https://ultralytics.com/images/bus.jpg")
        #     >>> for r in results:
        #     ...     print(f"Detected {len(r)} objects in image")
        return self.predict(source, stream, **kwargs)  # predictメソッドを呼び出す

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        # 新しいモデルを初期化し、モデル定義からタスクタイプを推測します。
        # このメソッドは、提供された設定ファイルに基づいて新しいモデルインスタンスを作成します。
        # モデル設定をロードし、タスクタイプが指定されていない場合は推測し、タスクマップから適切なクラスを使用してモデルを初期化します。
        #
        # 引数:
        #     cfg (str): YAML形式のモデル設定ファイルへのパス。
        #     task (str | None): モデルの特定のタスク。Noneの場合、設定から推測されます。
        #     model (torch.nn.Module | None): カスタムモデルインスタンス。提供されている場合は、新しいモデルを作成する代わりに使用されます。
        #     verbose (bool): Trueの場合、ロード中にモデル情報を表示します。
        #
        # 例外:
        #     ValueError: 設定ファイルが無効であるか、タスクを推測できない場合。
        #     ImportError: 指定されたタスクに必要な依存関係がインストールされていない場合。
        #
        # 例:
        #     >>> model = Model()
        #     >>> model._new("yolov8n.yaml", task="detect", verbose=True)
        cfg_dict = yaml_model_load(cfg)  # YAMLファイルから設定をロード
        self.cfg = cfg  # 設定を保存
        self.task = task or guess_model_task(cfg_dict)  # タスクを推測
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # モデルを構築
        self.overrides["model"] = self.cfg  # 上書き設定を更新
        self.overrides["task"] = self.task  # 上書き設定を更新

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task  # モデルのタスクを設定
        self.model_name = cfg  # モデル名を設定

    def _load(self, weights: str, task=None) -> None:
        # チェックポイントファイルからモデルをロードするか、重みファイルから初期化します。
        # このメソッドは、.ptチェックポイントファイルまたは他の重みファイル形式からのモデルのロードを処理します。
        # ロードされた重みに基づいて、モデル、タスク、および関連する属性を設定します。
        #
        # 引数:
        #     weights (str): ロードするモデルの重みファイルへのパス。
        #     task (str | None): モデルに関連付けられたタスク。Noneの場合、モデルから推測されます。
        #
        # 例外:
        #     FileNotFoundError: 指定された重みファイルが存在しないか、アクセスできない場合。
        #     ValueError: 重みファイル形式がサポートされていないか、無効な場合。
        #
        # 例:
        #     >>> model = Model()
        #     >>> model._load("yolo11n.pt")
        #     >>> model._load("path/to/weights.pth", task="detect")
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):  # ウェイトがURLから始まる場合
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file。ローカルファイルをダウンロードして返す
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolov8n -> yolov8n.pt。サフィックスを追加する（例：yolov8n -> yolov8n.pt）

        if Path(weights).suffix == ".pt":  # ウェイトファイルの拡張子が.ptの場合
            self.model, self.ckpt = attempt_load_one_weight(weights)  # モデルとチェックポイントをロード
            self.task = self.model.args["task"]  # タスクをロード
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)  # 上書き設定をリセット
            self.ckpt_path = self.model.pt_path  # チェックポイントパスをロード
        else:  # ウェイトファイルの拡張子が.ptではない場合
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call。すべての場合に実行
            self.model, self.ckpt = weights, None  # モデルとチェックポイントをウェイトとNoneに設定
            self.task = task or guess_model_task(weights)  # タスクを推測
            self.ckpt_path = weights  # チェックポイントパスをウェイトに設定
        self.overrides["model"] = weights  # モデルの上書き設定をウェイトに設定
        self.overrides["task"] = self.task  # タスクの上書き設定をタスクに設定
        self.model_name = weights  # モデル名をウェイトに設定

    def _check_is_pytorch_model(self) -> None:
        # モデルがPyTorchモデルであるかどうかを確認し、そうでない場合はTypeErrorを発生させます。
        # このメソッドは、モデルがPyTorchモジュールまたは.ptファイルのいずれかであることを確認します。
        # PyTorchモデルを必要とする特定の操作が、互換性のあるモデルタイプでのみ実行されるようにするために使用されます。
        #
        # 例外:
        #     TypeError: モデルがPyTorchモジュールまたは.ptファイルではない場合。
        #         エラーメッセージは、サポートされているモデル形式と操作に関する詳細情報を提供します。
        #
        # 例:
        #     >>> model = Model("yolo11n.pt")
        #     >>> model._check_is_pytorch_model()  # エラーは発生しません
        #     >>> model = Model("yolov8n.onnx")
        #     >>> model._check_is_pytorch_model()  # TypeErrorが発生します
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"  # モデルが文字列またはPath型で、拡張子が.ptかどうか
        pt_module = isinstance(self.model, nn.Module)  # モデルがnn.Moduleのインスタンスかどうか
        if not (pt_module or pt_str):  # PyTorchモジュールまたは.ptファイルでない場合
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":
        # モデルの重みを初期状態にリセットします。
        # このメソッドは、モデル内のすべてのモジュールを反復処理し、'reset_parameters'メソッドがある場合はそのパラメータをリセットします。
        # また、すべてのパラメータの'requires_grad'がTrueに設定されていることを確認し、トレーニング中に更新できるようにします。
        #
        # 戻り値:
        #     (Model): 重みをリセットしたクラスのインスタンス。
        #
        # 例外:
        #     AssertionError: モデルがPyTorchモデルではない場合。
        #
        # 例:
        #     >>> model = Model("yolo11n.pt")
        #     >>> model.reset_weights()
        self._check_is_pytorch_model()  # PyTorchモデルかどうかチェック
        for m in self.model.modules():  # モデル内のすべてのモジュールを反復処理
            if hasattr(m, "reset_parameters"):  # モジュールがreset_parameters属性を持っている場合
                m.reset_parameters()  # パラメータをリセット
        for p in self.model.parameters():  # モデル内のすべてのパラメータを反復処理
            p.requires_grad = True  # requires_gradをTrueに設定
        return self  # 自身のインスタンスを返す

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> "Model":
        # 指定された重みファイルからパラメータをモデルにロードします。
        # このメソッドは、ファイルから、または重みオブジェクトから直接重みをロードすることをサポートします。
        # パラメータを名前と形状で照合し、モデルに転送します。
        #
        # 引数:
        #     weights (Union[str, Path]): 重みファイルまたは重みオブジェクトへのパス。
        #
        # 戻り値:
        #     (Model): 重みをロードしたクラスのインスタンス。
        #
        # 例外:
        #     AssertionError: モデルがPyTorchモデルではない場合。
        #
        # 例:
        #     >>> model = Model()
        #     >>> model.load("yolo11n.pt")
        #     >>> model.load(Path("path/to/weights.pt"))
        self._check_is_pytorch_model()  # PyTorchモデルかどうかチェック
        if isinstance(weights, (str, Path)):  # ウェイトが文字列またはPath型の場合
            self.overrides["pretrained"] = weights  # remember the weights for DDP training。DDPトレーニングのためにウェイトを記憶
            weights, self.ckpt = attempt_load_one_weight(weights)  # モデルとチェックポイントをロード
        self.model.load(weights)  # モデルにウェイトをロード
        return self  # 自身のインスタンスを返す

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        # 現在のモデルの状態をファイルに保存します。
        # このメソッドは、モデルのチェックポイント（ckpt）を指定されたファイル名にエクスポートします。
        # 日付、Ultralyticsバージョン、ライセンス情報、およびドキュメントへのリンクなどのメタデータが含まれています。
        #
        # 引数:
        #     filename (Union[str, Path]): モデルの保存先のファイルの名前。
        #
        # 例外:
        #     AssertionError: モデルがPyTorchモデルではない場合。
        #
        # 例:
        #     >>> model = Model("yolo11n.pt")
        #     >>> model.save("my_model.pt")
        self._check_is_pytorch_model()  # PyTorchモデルかどうかチェック
        from copy import deepcopy  # 深いコピーを行うためのモジュールをインポート
        from datetime import datetime  # 日付と時刻を扱うためのモジュールをインポート

        from ultralytics import __version__  # Ultralyticsのバージョン情報をインポート

        updates = {  # 保存する情報の辞書
            "model": deepcopy(self.model).half() if isinstance(self.model, nn.Module) else self.model,  # モデルのコピー（nn.Moduleの場合、半精度に変換）
            "date": datetime.now().isoformat(),  # 現在の日付と時刻
            "version": __version__,  # Ultralyticsのバージョン
           "license": "AGPL-3.0 License (https://ultralytics.com/license)",  # ライセンス：AGPL-3.0 License（https://ultralytics.com/license）
            "docs": "https://docs.ultralytics.com",  # ドキュメント：https://docs.ultralytics.com
        }
        torch.save({**self.ckpt, **updates}, filename)  # チェックポイント(self.ckpt)と更新情報(updates)をfilenameに保存

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        モデル情報をログ出力または返します。

        このメソッドは、渡された引数に応じて、モデルの概要または詳細情報を提供します。
        出力の冗長性を制御し、情報をリストとして返すことができます。

        Args:
            detailed (bool): Trueの場合、モデルのレイヤーとパラメータに関する詳細情報を表示します。
            verbose (bool): Trueの場合、情報を出力します。 Falseの場合、情報をリストとして返します。

        Returns:
            (List[str]): モデルの概要、レイヤーの詳細、パラメータ数など、モデルに関するさまざまな種類の情報を含む文字列のリスト。
            verboseがTrueの場合は空です。

        Raises:
            TypeError: モデルがPyTorchモデルでない場合。

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # モデルの概要を出力
            >>> info_list = model.info(detailed=True, verbose=False)  # 詳細情報をリストとして返す
        """
        self._check_is_pytorch_model()  # モデルがPyTorchモデルであることを確認
        return self.model.info(detailed=detailed, verbose=verbose)  # モデル情報を取得して返す

    def fuse(self):
        """
        最適化された推論のために、モデル内のConv2dレイヤーとBatchNorm2dレイヤーを融合します。

        このメソッドは、モデルのモジュールを反復処理し、連続するConv2dレイヤーとBatchNorm2dレイヤーを単一のレイヤーに融合します。
        この融合により、順方向パス中に必要な操作とメモリアクセスの回数を減らすことで、推論速度を大幅に向上させることができます。

        融合プロセスには通常、BatchNorm2dパラメータ（平均、分散、重み、バイアス）を前のConv2dレイヤーの重みとバイアスに畳み込むことが含まれます。
        これにより、畳み込みと正規化を1つのステップで実行する単一のConv2dレイヤーが得られます。

        Raises:
            TypeError: モデルがPyTorch nn.Moduleでない場合。

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # モデルが融合され、最適化された推論の準備ができました
        """
        self._check_is_pytorch_model()  # モデルがPyTorchモデルであることを確認
        self.model.fuse()  # モデルのConv2dレイヤーとBatchNorm2dレイヤーを融合

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,  # 入力ソース。ファイルパス、URL、PILイメージ、numpy配列、torchテンソルなど
        stream: bool = False,  # ストリーミングモードで処理するかどうか。デフォルトはFalse
        **kwargs,  # その他のキーワード引数
    ) -> list:
        """
        指定されたソースに基づいて画像埋め込みを生成します。

        このメソッドは`predict()`メソッドのラッパーであり、画像ソースから埋め込みを生成することに焦点を当てています。
        さまざまなキーワード引数を使用して、埋め込みプロセスをカスタマイズできます。

        Args:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor): 埋め込みを生成する画像のソース。
                ファイルパス、URL、PILイメージ、numpy配列などを指定できます。
            stream (bool): Trueの場合、予測はストリーミングされます。
            **kwargs (Any): 埋め込みプロセスを構成するための追加のキーワード引数。

        Returns:
            (List[torch.Tensor]): 画像埋め込みを含むリスト。

        Raises:
            AssertionError: モデルがPyTorchモデルでない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):  # 'embed'キーがkwargsに存在しない場合
            kwargs["embed"] = [len(self.model.model) - 2]  # インデックスが渡されなかった場合、最後から2番目のレイヤーを埋め込みます。
        return self.predict(source, stream, **kwargs)  # predictメソッドを呼び出し、埋め込みを生成

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,  # 入力ソース。ファイルパス、URL、PILイメージ、numpy配列、torchテンソルなど
        stream: bool = False,  # ストリーミングモードで処理するかどうか。デフォルトはFalse
        predictor=None,  # カスタムpredictor。デフォルトはNone
        **kwargs,  # その他のキーワード引数
    ) -> List[Results]:
        """
        YOLOモデルを使用して、指定された画像ソースで予測を実行します。

        このメソッドは、キーワード引数を使用してさまざまな構成を可能にし、予測プロセスを容易にします。
        カスタムpredictorまたはデフォルトのpredictorメソッドを使用した予測をサポートします。
        このメソッドは、さまざまなタイプの画像ソースを処理し、ストリーミングモードで動作できます。

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): 予測を行う画像のソース。
                ファイルパス、URL、PILイメージ、numpy配列、torchテンソルなどのさまざまなタイプを受け入れます。
            stream (bool): Trueの場合、入力ソースを予測用の連続ストリームとして扱います。
            predictor (BasePredictor | None): 予測を行うためのカスタムpredictorクラスのインスタンス。
                Noneの場合、メソッドはデフォルトのpredictorを使用します。
            **kwargs (Any): 予測プロセスを構成するための追加のキーワード引数。

        Returns:
            (List[ultralytics.engine.results.Results]): 予測結果のリスト。各結果はResultsオブジェクトにカプセル化されます。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # 検出されたバウンディングボックスを出力

        Notes:
            - 「source」が指定されていない場合、警告とともにデフォルトでASSETS定数になります。
            - このメソッドは、新しいpredictorがまだ存在しない場合にセットアップし、各呼び出しでその引数を更新します。
            - SAMタイプのモデルの場合、「prompts」をキーワード引数として渡すことができます。
        """
        if source is None:  # ソースが指定されていない場合
            source = ASSETS  # ソースをアセットに設定
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")  # 警告メッセージを記録

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(  # CLI環境かどうかをチェック
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # メソッドのデフォルト値
        args = {**self.overrides, **custom, **kwargs}  # 優先度の高い順に引数を結合
        prompts = args.pop("prompts", None)  # SAMタイプのモデル用のプロンプト

        if not self.predictor:  # predictorがまだ設定されていない場合
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)  # predictorをロード
            self.predictor.setup_model(model=self.model, verbose=is_cli)  # predictorをセットアップ
        else:  # predictorがすでに設定されている場合のみ、引数を更新
            self.predictor.args = get_cfg(self.predictor.args, args) # predictorの引数を更新
            if "project" in args or "name" in args: # projectまたはname引数が渡された場合
                self.predictor.save_dir = get_save_dir(self.predictor.args) # 保存ディレクトリを更新

        if prompts and hasattr(self.predictor, "set_prompts"):  # SAMタイプのモデルの場合、プロンプトを設定
            self.predictor.set_prompts(prompts) # プロンプトを設定

        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)  # 予測を実行
    
    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,  # 入力ソース：ファイルパス、URL、PILイメージ、numpy配列、torchテンソルなど
        stream: bool = False,  # ストリーミング入力かどうか。デフォルトはFalse
        persist: bool = False,  # トラッカーを永続化するかどうか。デフォルトはFalse
        **kwargs,  # その他のキーワード引数
    ) -> List[Results]:
        """
        登録されたトラッカーを使用して、指定された入力ソースでオブジェクト追跡を実行します。

        このメソッドは、モデルの予測子とオプションで登録されたトラッカーを使用してオブジェクト追跡を実行します。
        ファイルパスやビデオストリームなどのさまざまな入力ソースを処理し、キーワード引数によるカスタマイズをサポートします。
        メソッドは、トラッカーがまだ存在しない場合に登録し、呼び出し間でそれらを永続化できます。

        Args:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], optional): オブジェクト追跡の入力ソース。
                ファイルパス、URL、またはビデオストリームを指定できます。デフォルトはNoneです。
            stream (bool): Trueの場合、入力ソースを連続ビデオストリームとして扱います。デフォルトはFalseです。
            persist (bool): Trueの場合、このメソッドの異なる呼び出し間でトラッカーを永続化します。デフォルトはFalseです。
            **kwargs (Any): 追跡プロセスを構成するための追加のキーワード引数。

        Returns:
            (List[ultralytics.engine.results.Results]): 追跡結果のリスト。各結果はResultsオブジェクトです。

        Raises:
            AttributeError: 予測子が登録されたトラッカーを持っていない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # 追跡IDを出力

        Notes:
            - このメソッドは、ByteTrackベースの追跡に対してデフォルトの信頼度しきい値0.1を設定します。
            - 追跡モードは、キーワード引数で明示的に設定されます。
            - バッチサイズは、ビデオの追跡用に1に設定されます。
        """
        if not hasattr(self.predictor, "trackers"):  # predictorがトラッカーを持っていない場合
            from trackers import register_tracker  # トラッカーを登録する関数をインポート

            register_tracker(self, persist)  # トラッカーを登録
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrackベースのメソッドは、入力として低い信頼度の予測を必要とします
        kwargs["batch"] = kwargs.get("batch") or 1  # ビデオでの追跡の場合、バッチサイズは1
        kwargs["mode"] = "track"  # 追跡モードを設定
        return self.predict(source=source, stream=stream, **kwargs)  # 予測を実行して追跡結果を返す

    def val(
        self,
        validator=None,  # カスタムvalidator。デフォルトはNone
        **kwargs,  # その他のキーワード引数
    ):
        """
        指定されたデータセットと検証構成を使用してモデルを検証します。

        このメソッドは、さまざまな設定によるカスタマイズを可能にし、モデル検証プロセスを容易にします。
        カスタムバリデーターまたはデフォルトの検証アプローチによる検証をサポートします。
        このメソッドは、デフォルト構成、メソッド固有のデフォルト、およびユーザーが指定した引数を組み合わせて、検証プロセスを構成します。

        Args:
            validator (ultralytics.engine.validator.BaseValidator | None): モデルを検証するためのカスタムバリデータークラスのインスタンス。
            **kwargs (Any): 検証プロセスをカスタマイズするための任意のキーワード引数。

        Returns:
            (ultralytics.utils.metrics.DetMetrics): 検証プロセスから取得された検証メトリック。

        Raises:
            AssertionError: モデルがPyTorchモデルでない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # mAP50-95を出力
        """
        custom = {"rect": True}  # method defaults。メソッドのデフォルト引数
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right。引数を結合

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)  # validatorを初期化
        validator(model=self.model)  # モデルを検証
        self.metrics = validator.metrics  # メトリックを保存
        return validator.metrics  # 検証メトリックを返す

    def benchmark(
        self,
        **kwargs,  # その他のキーワード引数
    ):
        """
        さまざまなエクスポート形式でモデルをベンチマークして、パフォーマンスを評価します。

        このメソッドは、ONNX、TorchScriptなどのさまざまなエクスポート形式でのモデルのパフォーマンスを評価します。
        ultralytics.utils.benchmarksモジュールの「benchmark」関数を使用します。
        ベンチマークは、デフォルトの構成値、モデル固有の引数、メソッド固有のデフォルト、および追加のユーザー指定のキーワード引数の組み合わせを使用して構成されます。

        Args:
            **kwargs (Any): ベンチマークプロセスをカスタマイズするための任意のキーワード引数。これらは、デフォルト構成、モデル固有の引数、およびメソッドのデフォルトと組み合わされます。
                一般的なオプションは次のとおりです。
                - data (str): ベンチマーク用のデータセットへのパス。
                - imgsz (int | List[int]): ベンチマーク用の画像サイズ。
                - half (bool): 半精度（FP16）モードを使用するかどうか。
                - int8 (bool): int8精度モードを使用するかどうか。
                - device (str): ベンチマークを実行するデバイス（例：'cpu'、'cuda'）。
                - verbose (bool): 詳細なベンチマーク情報を出力するかどうか。

        Returns:
            (Dict): さまざまなエクスポート形式のメトリックを含む、ベンチマークプロセスの結果を含む辞書。

        Raises:
            AssertionError: モデルがPyTorchモデルでない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()  # モデルがPyTorchモデルであることを確認
        from utils.benchmarks import benchmark  # ベンチマーク関数をインポート

        custom = {"verbose": False}  # method defaults。メソッドのデフォルト引数
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}  # 引数を結合
        return benchmark(  # ベンチマークを実行
            model=self,
            data=kwargs.get("data"),  # 'data'引数が渡されない場合は、デフォルトのデータセットに対してdata = Noneを設定します。
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs,  # その他のキーワード引数
    ) -> str:
        """
        デプロイに適した別の形式にモデルをエクスポートします。

        このメソッドは、デプロイメントの目的でモデルをさまざまな形式（たとえば、ONNX、TorchScript）にエクスポートすることを容易にします。
        エクスポートプロセスには「Exporter」クラスを使用し、モデル固有のオーバーライド、メソッドのデフォルト、および追加の引数を組み合わせて使用​​します。

        Args:
            **kwargs (Dict): エクスポートプロセスをカスタマイズするための任意のキーワード引数。
            これらは、モデルのオーバーライドおよびメソッドのデフォルトと組み合わされます。
            一般的な引数は次のとおりです。
                format (str): エクスポート形式（例：「onnx」、「エンジン」、「coreml」）。
                half (bool): 半精度でモデルをエクスポートします。
                int8 (bool): int8精度でモデルをエクスポートします。
                device (str): エクスポートを実行するデバイス。
                workspace (int): TensorRTエンジンの最大メモリワークスペースサイズ。
                nms (bool): モデルにNon-Maximum Suppression（NMS）モジュールを追加します。
                simplify (bool): ONNXモデルを簡略化します。

        Returns:
            (str): エクスポートされたモデルファイルへのパス。

        Raises:
            AssertionError: モデルがPyTorchモデルでない場合。
            ValueError: サポートされていないエクスポート形式が指定されている場合。
            RuntimeError: エラーが原因でエクスポートプロセスが失敗した場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()  # モデルがPyTorchモデルであることを確認
        from .exporter import Exporter  # Exporterクラスをインポート

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # マルチGPUエラーを回避するためにリセット
            "verbose": False,
        }  # メソッドのデフォルト
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # 優先度の高い順に引数を結合
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)  # モデルをエクスポート

    def train(
        self,
        trainer=None,  # カスタムtrainer。デフォルトはNone
        **kwargs,  # その他のキーワード引数
    ):
        """
        指定されたデータセットとトレーニング構成を使用してモデルをトレーニングします。

        このメソッドは、カスタマイズ可能なさまざまな設定でモデルトレーニングを容易にします。
        カスタムトレーナーまたはデフォルトのトレーニングアプローチによるトレーニングをサポートします。
        このメソッドは、チェックポイントからのトレーニングの再開、Ultralytics HUBとの統合、およびトレーニング後のモデルと構成の更新などのシナリオを処理します。

        Ultralytics HUBを使用する場合、セッションにロードされたモデルがある場合、このメソッドはHUBトレーニング引数を優先し、ローカル引数が指定された場合は警告します。
        pipアップデートを確認し、デフォルト構成、メソッド固有のデフォルト、およびユーザーが指定した引数を組み合わせて、トレーニングプロセスを構成します。

        Args:
            trainer (BaseTrainer | None): モデルトレーニング用のカスタムトレーナーインスタンス。 Noneの場合、デフォルトを使用します。
            **kwargs (Any): トレーニング構成用の任意のキーワード引数。一般的なオプションは次のとおりです。
                data (str): データセット構成ファイルへのパス。
                epochs (int): トレーニングエポック数。
                batch_size (int): トレーニングのバッチサイズ。
                imgsz (int): 入力画像サイズ。
                device (str): トレーニングを実行するデバイス（例：'cuda'、'cpu'）。
                workers (int): データロード用のワーカースレッドの数。
                optimizer (str): トレーニングに使用するオプティマイザ。
                lr0 (float): 初期学習率。
                patience (int): トレーニングの早期停止のために観察可能な改善がない場合に待機するエポック。

        Returns:
            (Dict | None): 使用可能で、トレーニングが成功した場合のトレーニングメトリック。それ以外の場合はNone。

        Raises:
            AssertionError: モデルがPyTorchモデルでない場合。
            PermissionError: HUBセッションに権限の問題がある場合。
            ModuleNotFoundError: HUB SDKがインストールされていない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
        """
        self._check_is_pytorch_model()  # モデルがPyTorchモデルであることを確認
        if hasattr(self.session, "model") and self.session.model.id:  # ロードされたモデルを使用したUltralytics HUBセッション
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ HUBトレーニング引数を使用しており、ローカルトレーニング引数は無視されます。")  # 警告メッセージを記録
            kwargs = self.session.train_args  # kwargsを上書き

        checks.check_pip_update_available()  # pipのアップデートを確認

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides  # 設定ファイルをロード
        custom = {
            # NOTE: 'cfg'に'data'が含まれている場合の処理。
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],  # データセットの設定
            "model": self.overrides["model"],  # モデルを設定
            "task": self.task,  # タスクを設定
        }  # メソッドのデフォルト
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # 優先度の高い順に引数を結合
        if args.get("resume"):  # トレーニングを再開する場合
            args["resume"] = self.ckpt_path  # チェックポイントのパスを設定

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)  # trainerを初期化
        if not args.get("resume"):  # 手動でモデルを設定（再開しない場合のみ）
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)  # モデルを取得
            self.model = self.trainer.model  # モデルを設定

        self.trainer.hub_session = self.session  # オプションのHUBセッションをアタッチ
        self.trainer.train()  # トレーニングを開始
        # トレーニング後にモデルとcfgを更新
        if RANK in {-1, 0}:  # ランクが-1または0の場合
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last  # 最高のチェックポイントを取得
            self.model, _ = attempt_load_one_weight(ckpt)  # モデルをロード
            self.overrides = self.model.args  # オーバーライドを設定
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # メトリックを取得
        return self.metrics  # メトリックを返す

    def tune(
        self,
        use_ray=False,  # Ray Tuneを使用するかどうか。デフォルトはFalse
        iterations=10,  # チューニングの反復回数。デフォルトは10
        *args,  # 追加の引数
        **kwargs,  # その他のキーワード引数
    ):
        """
        Ray Tuneを使用するオプションを使用して、モデルのハイパーパラメータチューニングを実行します。

        このメソッドは、ハイパーパラメータチューニングの2つのモードをサポートしています。Ray Tuneの使用、またはカスタムチューニングメソッドの使用です。
        Ray Tuneが有効になっている場合、ultralytics.utils.tunerモジュールの「run_ray_tune」関数を利用します。
        それ以外の場合は、内部の「Tuner」クラスを使用してチューニングします。
        このメソッドは、デフォルト、オーバーライド、およびカスタム引数を組み合わせて、チューニングプロセスを構成します。

        Args:
            use_ray (bool): Trueの場合、ハイパーパラメータチューニングにRay Tuneを使用します。デフォルトはFalseです。
            iterations (int): 実行するチューニングの反復回数。デフォルトは10です。
            *args (List): 追加の引数の可変長引数リスト。
            **kwargs (Dict): 任意のキーワード引数。これらは、モデルのオーバーライドおよびデフォルトと組み合わされます。

        Returns:
            (Dict): ハイパーパラメータ検索の結果を含む辞書。

        Raises:
            AssertionError: モデルがPyTorchモデルでない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(use_ray=True, iterations=20)
            >>> print(results)
        """
        self._check_is_pytorch_model()  # モデルがPyTorchモデルであることを確認
        if use_ray:  # Ray Tuneを使用する場合
            from utils.tuner import run_ray_tune  # run_ray_tune関数をインポート

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)  # Ray Tuneを実行
        else:  # Ray Tuneを使用しない場合
            from .tuner import Tuner  # Tunerクラスをインポート

            custom = {}  # method defaults。メソッドのデフォルト引数
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right。引数を結合
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)  # Tunerを実行

    def _apply(self, fn) -> "Model":
        """
        パラメータまたは登録されたバッファではないモデルテンソルに関数を適用します。

        このメソッドは、predictorをリセットし、モデルのオーバーライドでデバイスを更新することにより、親クラスの_applyメソッドの機能を拡張します。
        通常、モデルを別のデバイスに移動したり、その精度を変更したりするなどの操作に使用されます。

        Args:
            fn (Callable): モデルのテンソルに適用される関数。これは通常、to()、cpu()、cuda()、half()、float()などのメソッドです。

        Returns:
            (Model): 関数が適用され、属性が更新されたモデルインスタンス。

        Raises:
            AssertionError: モデルがPyTorchモデルでない場合。

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # モデルをGPUに移動
        """
        self._check_is_pytorch_model()  # モデルがPyTorchモデルであることを確認
        self = super()._apply(fn)  # noqa
        self.predictor = None  # デバイスが変更された可能性があるため、predictorをリセット
        self.overrides["device"] = self.device  # self.deviceの文字列表現(例: device(type='cuda', index=0)) -> 'cuda:0'
        return self  # 更新されたモデルインスタンスを返す

    @property
    def names(self) -> list:
        """
        ロードされたモデルに関連付けられたクラス名を取得します。

        このプロパティは、モデルで定義されている場合はクラス名を返します。
        ultralytics.nn.autobackendモジュールの「check_class_names」関数を使用して、クラス名の有効性を確認します。
        predictorが初期化されていない場合は、名前を取得する前にセットアップします。

        Returns:
            (Dict[int, str]): モデルに関連付けられたクラス名のdict。

        Raises:
            AttributeError: モデルまたはpredictorに「names」属性がない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from nn.autobackend import check_class_names  # check_class_names関数をインポート

        if hasattr(self.model, "names"):  # モデルにnames属性がある場合
            return check_class_names(self.model.names)  # クラス名をチェックして返す
        if not self.predictor:  # エクスポート形式では、predict()が呼び出されるまでpredictorが定義されません
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)  # predictorをロード
            self.predictor.setup_model(model=self.model, verbose=False)  # predictorをセットアップ
        return self.predictor.model.names  # predictorからクラス名を返す

    @property
    def device(self) -> torch.device:
        """
        モデルのパラメータが割り当てられているデバイスを取得します。

        このプロパティは、モデルのパラメータが現在格納されているデバイス（CPUまたはGPU）を決定します。
        これは、nn.Moduleのインスタンスであるモデルにのみ適用されます。

        Returns:
            (torch.device): モデルのデバイス（CPU / GPU）。

        Raises:
            AttributeError: モデルがPyTorch nn.Moduleインスタンスでない場合。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # CUDAが利用可能な場合
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None  # モデルのデバイスを返す

    @property
    def transforms(self):
        """
        ロードされたモデルの入力データに適用される変換を取得します。

        このプロパティは、モデルで定義されている場合は変換を返します。
        変換には通常、モデルに入力される前に入力データに適用される、サイズ変更、正規化、データ拡張などの前処理ステップが含まれます。

        Returns:
            (object | None): モデルで使用可能な場合は変換オブジェクト、それ以外の場合はNone。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None  # モデルの変換を返す

    def add_callback(self, event: str, func) -> None:
        """
        指定されたイベントのコールバック関数を追加します。

        このメソッドを使用すると、トレーニングや推論などのモデル操作中に特定のイベントでトリガーされるカスタムコールバック関数を登録できます。
        コールバックは、モデルのライフサイクルのさまざまな段階でモデルの動作を拡張およびカスタマイズする方法を提供します。

        Args:
            event (str): コールバックをアタッチするイベントの名前。Ultralyticsフレームワークで認識される有効なイベント名である必要があります。
            func (Callable): 登録するコールバック関数。この関数は、指定されたイベントが発生したときに呼び出されます。

        Raises:
            ValueError: イベント名が認識されないか、無効な場合。

        Examples:
            >>> def on_train_start(trainer):
            ...     print("トレーニングが開始されます！")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)  # コールバック関数を追加

    def clear_callback(self, event: str) -> None:
        """
        指定されたイベントに登録されているすべてのコールバック関数をクリアします。

        このメソッドは、指定されたイベントに関連付けられているすべてのカスタムおよびデフォルトのコールバック関数を削除します。
        指定されたイベントのコールバックリストを空のリストにリセットし、そのイベントに登録されているすべてのコールバックを効果的に削除します。

        Args:
            event (str): コールバックをクリアするイベントの名前。これは、Ultralyticsコールバックシステムで認識される有効なイベント名である必要があります。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("トレーニングが開始されました"))
            >>> model.clear_callback("on_train_start")
            >>> # 'on_train_start'のすべてのコールバックが削除されました

        Notes:
            - このメソッドは、ユーザーが追加したカスタムコールバックと、Ultralyticsフレームワークによって提供されるデフォルトのコールバックの両方に影響します。
            - このメソッドを呼び出した後、新しいコールバックが追加されるまで、指定されたイベントに対してコールバックは実行されません。
            - 特定の操作の適切な機能に必要な、重要なコールバックを含め、すべてのコールバックを削除するため、注意して使用してください。
        """
        self.callbacks[event] = []  # コールバックリストをクリア

    def reset_callbacks(self) -> None:
        """
        すべてのコールバックをデフォルト関数にリセットします。

        このメソッドは、すべてのイベントのデフォルトのコールバック関数を復元し、以前に追加されたカスタムコールバックを削除します。
        すべてのデフォルトのコールバックイベントを反復処理し、現在のコールバックをデフォルトのコールバックに置き換えます。

        デフォルトのコールバックは、「callbacks.default_callbacks」辞書で定義されています。これには、on_train_start、on_epoch_endなど、モデルのライフサイクルのさまざまなイベントに対する定義済みの関数が含まれています。

        このメソッドは、カスタム変更を加えた後で元のコールバックセットに戻し、異なる実行または実験間で一貫した動作を確保する場合に役立ちます。

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # すべてのコールバックがデフォルト関数にリセットされました
        """
        for event in callbacks.default_callbacks.keys():  # デフォルトのコールバックイベントを反復処理
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]  # コールバックをデフォルトにリセット

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        PyTorchモデルのチェックポイントをロードするときに、特定の引数をリセットします。

        この静的メソッドは、入力引数の辞書をフィルタリングして、モデルのロードに重要と見なされる特定のキーのセットのみを保持します。
        チェックポイントからモデルをロードするときに、関連する引数のみが保持されるようにするために使用されます。
        不要な、または潜在的に競合する設定は破棄します。

        Args:
            args (dict): さまざまなモデル引数と設定を含む辞書。

        Returns:
            (dict): 入力引数から指定されたincludeキーのみを含む新しい辞書。

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # PyTorchモデルをロードするときに、これらの引数のみを記憶します
        return {k: v for k, v in args.items() if k in include}  # 指定されたキーのみを含む辞書を返す

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
        """
        モデルタスクに基づいて適切なモジュールをロードします。

        このメソッドは、モデルの現在のタスクと指定されたキーに基づいて、適切なモジュール（モデル、トレーナー、バリデーター、または予測子）を動的に選択して返します。
        task_map属性を使用して、ロードする正しいモジュールを決定します。

        Args:
            key (str): ロードするモジュールのタイプ。 「model」、「trainer」、「validator」、「predictor」のいずれかである必要があります。

        Returns:
            (object): 指定されたキーと現在のタスクに対応するロードされたモジュール。

        Raises:
            NotImplementedError: 指定されたキーが現在のタスクでサポートされていない場合。

        Examples:
            >>> model = Model(task="detect")
            >>> predictor = model._smart_load("predictor")
            >>> trainer = model._smart_load("trainer")

        Notes:
            - このメソッドは通常、Modelクラスの他のメソッドによって内部的に使用されます。
            - task_map属性は、各タスクに対して正しいマッピングで適切に初期化する必要があります。
        """
        try:
            return self.task_map[self.task][key]  # タスクマップからモジュールをロード
        except Exception as e:  # エラーが発生した場合
            name = self.__class__.__name__  # クラス名を取得
            mode = inspect.stack()[1][3]  # 関数名を取得
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}'モデルは、'{self.task}'タスクに対して'{mode}'モードをまだサポートしていません。")
            ) from e  # NotImplementedErrorを発生

    @property
    def task_map(self) -> dict:
        """
        モデルタスクからさまざまなモードに対応するクラスへのマッピングを提供します。

        このプロパティメソッドは、サポートされている各タスク（例：検出、セグメント、分類）をネストされた辞書にマッピングする辞書を返します。
        ネストされた辞書には、異なる操作モード（モデル、トレーナー、バリデーター、予測子）からそれぞれのクラス実装へのマッピングが含まれています。

        このマッピングにより、モデルのタスクと目的の操作モードに基づいて適切なクラスを動的にロードできます。
        これにより、Ultralyticsフレームワーク内のさまざまなタスクとモードを処理するための柔軟で拡張可能なアーキテクチャが容易になります。

        Returns:
            (Dict[str, Dict[str, Any]]): キーがタスク名（str）で、値がネストされた辞書である辞書。
            各ネストされた辞書には、「model」、「trainer」、「validator」、および「predictor」というキーがあり、それぞれのクラス実装にマッピングされます。

        Examples:
            >>> model = Model()
            >>> task_map = model.task_map
            >>> detect_class_map = task_map["detect"]
            >>> segment_class_map = task_map["segment"]

        Note:
            このメソッドの実際の実装は、Ultralyticsフレームワークでサポートされている特定のタスクとクラスによって異なる場合があります。
            ドキュメント文字列は、予想される動作と構造の一般的な説明を提供します。
        """
        raise NotImplementedError("モデルのタスクマップを提供してください!")  # task_mapが実装されていない場合、NotImplementedErrorを発生