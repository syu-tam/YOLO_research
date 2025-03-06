# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    v10Detect,
    Detectv2,
    Conv_withoutBN
)
from utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from utils.checks import check_requirements, check_suffix, check_yaml
from utils.loss import (
    E2EDetectLoss,
    v8DetectionLoss,
)
from utils.ops import make_divisible
from utils.plotting import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""
    # BaseModelクラスは、Ultralytics YOLOファミリーのすべてのモデルの基本クラスとして機能します。

    def forward(self, x, *args, **kwargs):
        # トレーニングまたは推論のいずれかのために、モデルの順方向パスを実行します。
        #
        # xがdictの場合、トレーニングの損失を計算して返します。それ以外の場合は、推論の予測を返します。
        #
        # 引数：
        #     x (torch.Tensor | dict): 推論の入力テンソル、またはトレーニングの画像テンソルとラベルを含むdict。
        #     *args (Any): 可変長引数リスト。
        #     **kwargs (Any): 任意のキーワード引数。
        #
        # 戻り値：
        #     (torch.Tensor): xがdict（トレーニング）の場合は損失、またはネットワーク予測（推論）。
        if isinstance(x, dict):  # for cases of training and validating while training。トレーニングおよびトレーニング中の検証の場合
            return self.loss(x, *args, **kwargs)  # 損失を計算して返す
        return self.predict(x, *args, **kwargs)  # 予測を計算して返す

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        # ネットワークを介して順方向パスを実行します。
        #
        # 引数：
        #     x (torch.Tensor): モデルへの入力テンソル。
        #     profile (bool): Trueの場合、各レイヤーの計算時間を出力します。デフォルトはFalseです。
        #     visualize (bool): Trueの場合、モデルのフィーチャマップを保存します。デフォルトはFalseです。
        #     augment (bool): 予測中に画像を拡張します。デフォルトはFalseです。
        #     embed (list, optional): 返すフィーチャベクトル/埋め込みのリスト（オプション）。
        #
        # 戻り値：
        #     (torch.Tensor): モデルの最後の出力。
        if augment:  # 拡張する場合
            return self._predict_augment(x)  # 拡張された予測を実行
        return self._predict_once(x, profile, visualize, embed)  # 一度の予測を実行

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        # ネットワークを介して順方向パスを実行します。
        #
        # 引数：
        #     x (torch.Tensor): モデルへの入力テンソル。
        #     profile (bool): Trueの場合、各レイヤーの計算時間を出力します。デフォルトはFalseです。
        #     visualize (bool): Trueの場合、モデルのフィーチャマップを保存します。デフォルトはFalseです。
        #     embed (list, optional): 返すフィーチャベクトル/埋め込みのリスト（オプション）。
        #
        # 戻り値：
        #     (torch.Tensor): モデルの最後の出力。
        y, dt, embeddings = [], [], []  # outputs。出力、時間、埋め込み
        for m in self.model:  # モデルのレイヤーを反復処理
            if m.f != -1:  # if not from previous layer。前のレイヤーからのものではない場合
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers。以前のレイヤーから
            if profile:  # プロファイルする場合
                self._profile_one_layer(m, x, dt)  # レイヤーをプロファイル
            x = m(x)  # run。実行
            y.append(x if m.i in self.save else None)  # save output。出力を保存
            if visualize:  # 可視化する場合
                feature_visualization(x, m.type, m.i, save_dir=visualize)  # 特徴を可視化
            if embed and m.i in embed:  # 埋め込む場合
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten。平坦化
                if m.i == max(embed):  # 最大の埋め込みインデックスに達した場合
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)  # 平坦化された埋め込みを返す
        return x  # 最後の出力を返す

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        # 入力画像xに対して拡張を実行し、拡張された推論を返します。
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )  # 警告ログを出力
        return self._predict_once(x)  # 一度の予測を実行

    def _profile_one_layer(self, m, x, dt):
        # 特定の入力に対するモデルの単一レイヤーの計算時間とFLOPをプロファイルします。
        # 結果を提供されたリストに追加します。
        #
        # 引数：
        #     m (nn.Module): プロファイルするレイヤー。
        #     x (torch.Tensor): レイヤーへの入力データ。
        #     dt (list): レイヤーの計算時間を格納するリスト。
        #
        # 戻り値：
        #     None
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix。最終レイヤーリストの場合、インプレース修正として入力をコピー
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs。GFLOPを計算
        t = time_sync()  # 現在時間を取得
        for _ in range(10):  # 10回繰り返す
            m(x.copy() if c else x)  # レイヤーを実行
        dt.append((time_sync() - t) * 100)  # 計算時間を追加
        if m == self.model[0]:  # 最初のレイヤーの場合
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")  # ログを出力
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")  # ログを出力
        if c:  # 最後のレイヤーの場合
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")  # ログを出力

    def fuse(self, verbose=True):
        # 計算効率を向上させるために、モデルの`Conv2d（）`レイヤーと`BatchNorm2d（）`レイヤーを単一のレイヤーに融合します。
        #
        # 戻り値：
        #     (nn.Module): 融合されたモデルが返されます。
        if not self.is_fused():  # 融合されていない場合
            for m in self.model.modules():  # モデルのモジュールを反復処理
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):  # Conv、Conv2、DWConvであり、bn属性がある場合
                    if isinstance(m, Conv2):  # Conv2の場合
                        m.fuse_convs()  # convを融合
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv。convを更新
                    delattr(m, "bn")  # remove batchnorm。バッチ正規化を削除
                    m.forward = m.forward_fuse  # update forward。順方向を更新
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):  # ConvTransposeであり、bn属性がある場合
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)  # 逆畳み込みを融合
                    delattr(m, "bn")  # remove batchnorm。バッチ正規化を削除
                    m.forward = m.forward_fuse  # update forward。順方向を更新
                if isinstance(m, RepConv):  # RepConvの場合
                    m.fuse_convs()  # convを融合
                    m.forward = m.forward_fuse  # update forward。順方向を更新
                if isinstance(m, RepVGGDW):  # RepVGGDWの場合
                    m.fuse()  # 融合
                    m.forward = m.forward_fuse  # 順方向を更新
            self.info(verbose=verbose)  # 情報を出力

        return self  # 自身を返す

    def is_fused(self, thresh=10):

        # モデルに、特定のしきい値よりも少ない数のBatchNormレイヤーがあるかどうかを確認します。
        #
        # 引数：
        #     thresh (int, optional): BatchNormレイヤーのしきい値数。デフォルトは10です。
        #
        # 戻り値：
        #     (bool): モデル内のBatchNormレイヤーの数がしきい値より少ない場合はTrue、それ以外の場合はFalse。
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()。正規化レイヤー
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model。モデルに「thresh」BatchNormレイヤーより少ない場合、True

    def info(self, detailed=False, verbose=True, imgsz=640):
        # モデル情報を出力します。
        #
        # 引数：
        #     detailed (bool): Trueの場合、モデルに関する詳細情報を出力します。デフォルトはFalse
        #     verbose (bool): Trueの場合、モデル情報を出力します。デフォルトはFalse
        #     imgsz (int): モデルのトレーニングに使用される画像のサイズ。デフォルトは640
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)  # モデル情報を出力

    def _apply(self, fn):
        # パラメータまたは登録されたバッファではない、モデル内のすべてのテンソルに関数を適用します。
        #
        # 引数：
        #     fn (function): モデルに適用する関数
        #
        # 戻り値：
        #     (BaseModel): 更新されたBaseModelオブジェクト。
        self = super()._apply(fn)  # 親クラスの_applyを呼び出し
        m = self.model[-1]  # Detect()。最後のレイヤーを取得
        if isinstance(m, Detect) or isinstance(m, Detectv2):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect。Detectサブクラスの場合
            m.stride = fn(m.stride)  # ストライドを更新
            m.anchors = fn(m.anchors)  # アンカーを更新
            m.strides = fn(m.strides)  # ストライドを更新
        return self  # 自身を返す

    def load(self, weights, verbose=True):
        # モデルに重みをロードします。
        #
        # 引数：
        #     weights (dict | torch.nn.Module): ロードする事前トレーニング済みの重み。
        #     verbose (bool, optional): 転送の進行状況を記録するかどうか。デフォルトはTrueです。
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts。torchvisionモデルはdictではありません
        csd = model.float().state_dict()  # checkpoint state_dict as FP32。FP32としてチェックポイントstate_dict
        csd = intersect_dicts(csd, self.state_dict())  # intersect。交差
        self.load_state_dict(csd, strict=False)  # load。ロード
        if verbose:  # 詳細モードの場合
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")  # ログを出力
        
    
    def loss(self, batch, preds=None):
        # 損失を計算します。
        #
        # 引数：
        #     batch (dict): 損失を計算するバッチ
        #     preds (torch.Tensor | List[torch.Tensor]): 予測。
        if getattr(self, "criterion", None) is None:  # 基準がない場合
            self.criterion = self.init_criterion()  # 基準を初期化
        preds = self.forward(batch["img"]) if preds is None else preds  # 予測を計算
        return self.criterion(preds, batch)  # 損失を計算

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        # BaseModelの損失基準を初期化します。
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")  # エラーを発生


class DetectionModel(BaseModel):
    
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.feature_maps = {
            'indices': {
                'pre_pan': [4, 6, 10],  # P3, P4, P5のインデックス
                'post_pan': [16, 19, 22]  # 1x1 Conv層のインデックス
            }
        }
    
        
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict。cfg dict
        if self.yaml["backbone"][0][2] == "Silence":  # バックボーンがサイレンスの場合
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )  # 警告ログを出力
            self.yaml["backbone"][0][2] = "nn.Identity"  # バックボーンをIdentityに変更

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels。入力チャンネル
        if nc and nc != self.yaml["nc"]:  # クラス数が異なる場合
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")  # ログを出力
            self.yaml["nc"] = nc  # override YAML value。YAML値をオーバーライド
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist。モデルを解析

        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict。デフォルトの名前辞書
        self.inplace = self.yaml.get("inplace", True)  # inplaceを設定
        self.end2end = getattr(self.model[-1], "end2end", False)  # end2endを設定

        # Build strides
        m = self.model[-1]  # Detect() ヘッドの定義
        if isinstance(m, Detect) or isinstance(m, Detectv2):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect。すべてのDetectサブクラスを含む
            s = 256  # 2x min stride。2倍の最小ストライド
            m.inplace = self.inplace  # inplaceを設定
            #ダミーのフォワード　
            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                # モデルを介して順方向パスを実行し、それに応じて異なるDetectサブクラスタイプを処理します。
                if self.end2end:  # end2endの場合
                    return self.forward(x)["one2many"]  # one2manyを返す
                return self.forward(x)  # 順方向パスを返す

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward。順方向パスを実行
            self.stride = m.stride  # ストライドを設定
            m.bias_init()  # only run once。一度だけ実行
        else:  # それ以外の場合
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR。デフォルトストライド

        # Init weights, biases
        initialize_weights(self)  # 重みを初期化
        if verbose:  # 詳細モードの場合
            self.info()  # 情報を表示
            LOGGER.info("")  # 空白行を出力
            

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Forward pass through model."""
        y, dt = [], []

        for i, m in enumerate(self.model):
            if isinstance(m, Detectv2):
                head_pre_pan_feats = [y[idx] for idx in self.feature_maps['indices']['pre_pan']]
                head_post_pan_feats = [y[idx] for idx in self.feature_maps['indices']['post_pan']]
                x = [head_pre_pan_feats, head_post_pan_feats]
                            

            elif m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # 順伝播
            x = m(x)
            
            y.append(x if m.i in (self.save or self.feature_maps['indices']['pre_pan'] or self.feature_maps['indices']['post_pan']) else None)
            
        return x
    
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        # 入力画像xで拡張を実行し、拡張された推論とトレーニング出力を返します。
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":  # end2endではない、またはクラス名がDetectionModelではない場合
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")  # 警告ログを出力
            return self._predict_once(x)  # 一度の予測を実行
        img_size = x.shape[-2:]  # height, width。高さと幅
        s = [1, 0.83, 0.67]  # scales。スケール
        f = [None, 3, None]  # flips (2-ud, 3-lr)。フリップ
        y = []  # outputs。出力
        for si, fi in zip(s, f):  # スケールとフリップを反復処理
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  # 画像をスケーリング
            yi = super().predict(xi)[0]  # forward。順方向パスを実行
            yi = self._descale_pred(yi, fi, si, img_size)  # デスケール
            y.append(yi)  # 結果を追加
        y = self._clip_augmented(y)  # clip augmented tails。拡張されたテールをクリップ
        return torch.cat(y, -1), None  # augmented inference, train。拡張された推論、トレーニング

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        # 拡張された推論（逆演算）に続いて予測をデスケールします。
        p[:, :4] /= scale  # de-scale。デスケール
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)  # 分割
        if flips == 2:  # udフリップの場合
            y = img_size[0] - y  # de-flip ud。udフリップを解除
        elif flips == 3:  # lrフリップの場合
            x = img_size[1] - x  # de-flip lr。lrフリップを解除
        return torch.cat((x, y, wh, cls), dim)  # 連結

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        # YOLO拡張推論テールをクリップします。
        nl = self.model[-1].nl  # number of detection layers (P3-P5)。検出レイヤー数
        g = sum(4**x for x in range(nl))  # grid points。グリッドポイント
        e = 1  # exclude layer count。除外レイヤー数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices。インデックス
        y[0] = y[0][..., :-i]  # large。大
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices。インデックス
        y[-1] = y[-1][..., i:]  # small。小
        return y  # 結果を返す


    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        # DetectionModelの損失基準を初期化します。
        #return E2EDetectLoss(self)
    
        if getattr(self, "end2end", False):  # end2endの場合
            return E2EDetectLoss(self)  # One2One Matching (E2E YOLO)。One2Oneマッチング（E2E YOLO）
        else:  # 通常の場合
            return v8DetectionLoss(self)  # 通常の YOLO 損失関数
        
    @property
    def epoch(self):
        return DetectionModel._epoch

    @epoch.setter
    def epoch(self, value):
        DetectionModel._epoch = value

    @property
    def total_epochs(self):
        return DetectionModel._total_epochs

    @total_epochs.setter
    def total_epochs(self,value):
        DetectionModel._total_epochs = value



class Ensemble(nn.ModuleList):
    """Ensemble of models."""
    # モデルのアンサンブル。

    def __init__(self):
        """Initialize an ensemble of models."""
        # モデルのアンサンブルを初期化します。
        super().__init__()  # 親クラスを初期化

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        # 関数はYOLOネットワークの最終レイヤーを生成します。
        y = [module(x, augment, profile, visualize)[0] for module in self]  # 各モデルの結果を取得
        # y = torch.stack(y).max(0)[0]  # max ensemble。最大アンサンブル
        # y = torch.stack(y).mean(0)  # mean ensemble。平均アンサンブル
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)。nmsアンサンブル
        return y, None  # inference, train output。推論、トレーニング出力

# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Example:
    ```python
    from nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    from utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """YOLOのmodel.yamlをPyTorchモデルに変換する関数"""
    import ast

    # 互換性のためのフラグ（v3/v5/v8/v9 のモデルをサポート）
    legacy = True  # 後方互換性確保
    max_channels = float("inf")  # 最大チャンネル数を無限大に設定
    
    # モデル設定の取得（存在しない場合はデフォルト値）
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))  # クラス数、活性化関数、スケール情報
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))  # スケール係数
    
    if scales:  # スケール情報が存在する場合
        scale = d.get("scale")  # スケール情報を取得
        if not scale:
            scale = tuple(scales.keys())[0]  # デフォルトのスケールを取得
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]  # スケールに応じてパラメータ調整

    if act:  # 活性化関数の設定
        Conv.default_act = eval(act)  # デフォルトの活性化関数を更新
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 設定をログ出力

    if verbose:  # 詳細情報の表示
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    
    ch = [ch]  # 入力チャンネルリスト
    layers, save, c2 = [], [], ch[-1]  # 各層のリスト、保存リスト、出力チャンネル
    
    # モデルのバックボーンとヘッド部分を構築
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # モジュール取得
        
        # 引数の文字列を適切な型に変換
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)  # 文字列から適切な型に変換
                except ValueError:
                    pass
        
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth倍率適用
        
        # 各モジュールごとのチャンネル数設定
        if m in {Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2fPSA,
                 C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN,
                 C2fAttn, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, PSA, SCDown,
                 C2fCIB, Conv_withoutBN}:
            c1, c2 = ch[f], args[0]  # 入出力チャンネル取得
            if c2 != nc:  # クラス数と異なる場合、適切なチャンネル数に調整
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # 特定モジュールの設定
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # 埋め込みチャネル数
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]  # 引数更新
            if m in {BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA}:
                args.insert(2, n)  # 繰り返し数を追加
                n = 1  # nを1にリセット
            if m in {C3k2}:
                legacy=False
                if scale in "mlx":
                    args[3] = True
             
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, ImagePoolingAttn, v10Detect,Detectv2}:
            if isinstance(f, list) and f and isinstance(f[0], list):
                args.append([[ch[x] for x in sub_f] for sub_f in f])
            else:
                args.append([ch[x] for x in f])
            if m is Detect:
                m.legacy = legacy
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]
        
        # モジュールのインスタンスを作成（繰り返し数に応じて処理）
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")  # モジュールの種類を文字列化
        m_.np = sum(x.numel() for x in m_.parameters())  # パラメータ数
        m_.i, m_.f, m_.type = i, f, t  # インデックスとタイプ情報を格納
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # ログ出力
        
        if isinstance(f, list):
            f_flat = list(flatten(f))
            if i != 0:
                save.extend(x % i for x in f_flat if x != -1)
            # もし i==0 の場合は何も追加しない
        else:
            if i != 0:
                save.append(f % i)
    # i==0 の場合は保存しない



        layers.append(m_)  # モジュール追加
        if i == 0:
            ch = []
        ch.append(c2)
    
    return nn.Sequential(*layers), sorted(save)  # 最終モデルを返す

def flatten(lst):
    """再帰的にリストを平坦化するジェネレータ"""
    for el in lst:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el



def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    # yolov11_e2e[nslmx].yaml -> yolov11_e2e.yaml の変換を追加
    match = re.match(r"(yolov\d+_e2e)([nslmx])?(.+)?$", path.stem)
    if match:
        new_stem = match.group(1)
        path = path.with_name(new_stem + path.suffix)


    unified_path = re.sub(r"(yolov\d+)(?:[nslmx])?(_e2e[nslmx]?)?(.+)?$", r"\1\3", str(path))

    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


import re
from pathlib import Path

def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x, or "" if not found.
    """
    try:
        match = re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem)
        if match:
            return match.group(1)
        
        # yolov11_e2es のようなケースに対応
        match = re.search(r"yolov\d+_e2e([nslmx])?", Path(model_path).stem)
        if match and match.group(1):
            return match.group(1)
            
        return "" # スケールが見つからない場合は空文字を返す
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        try:
            return cfg2task(model)
        except Exception:
            pass

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            try:
                return eval(x)["task"]
            except Exception:
                pass
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            try:
                return cfg2task(eval(x))
            except Exception:
                pass

        for m in model.modules():
            if isinstance(m, (Detect,v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
