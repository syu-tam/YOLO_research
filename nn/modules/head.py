# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn


from utils.tal import TORCH_1_10, dist2bbox, make_anchors

from .block import DFL
from .conv import Conv, DWConv, Conv_withoutBN


__all__ = "Detect", "v10Detect"

class Detect(nn.Module):
    """YOLO Detect head for detection models."""
    # 検出モデル用のYOLO検出ヘッド。

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        # 指定されたクラス数とチャネル数でYOLO検出レイヤーを初期化します。
        super().__init__()  # 親クラスを初期化
        self.nc = nc  # number of classes。クラス数
        self.nl = len(ch)  # number of detection layers。検出レイヤー数
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)。DFLチャンネル
        self.no = nc + self.reg_max * 4  # number of outputs per anchor。アンカーあたりの出力数
        self.stride = torch.zeros(self.nl)  # strides computed during build。ビルド中に計算されたストライド
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels。チャンネル
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )  # 畳み込みレイヤー
        
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )  # 畳み込みレイヤー
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  # DFLを設定

        if self.end2end:  # エンドツーエンドの場合
            self.one2one_cv2 = copy.deepcopy(self.cv2)  # cv2をコピー
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # cv3をコピー
        
        
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        # 予測されたバウンディングボックスとクラス確率を連結して返します。
        if self.end2end:  # エンドツーエンドの場合
            return self.forward_end2end(x)  # エンドツーエンドフォワードを実行
        
        for i in range(self.nl):  # 検出レイヤーを反復処理
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 特徴を連結
        if self.training:  # Training path。トレーニングパスの場合
            return x  # 特徴を返す
        y = self._inference(x)  # 推論を実行
            
        return y if self.export else (y, x)  # エクスポートまたは結果を返す


    def forward_end2end(self, x):
        # v10Detectモジュールの順方向パスを実行します。
        #
        # 引数：
        #     x (tensor): 入力テンソル。
        #
        # 戻り値：
        #     (dict, tensor): トレーニングモードでない場合は、one2many検出とone2one検出の両方の出力を含む辞書を返します。
        #         トレーニングモードの場合は、one2many検出とone2one検出の出力を別々に含む辞書を返します。
        x_detach = [xi.detach() for xi in x]  # 入力を分離
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]  # one2oneを計算

        for i in range(self.nl):  # レイヤーを反復処理
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 特徴を連結
        
        if self.training:  # Training path。トレーニングパスの場合
            return {"one2many": x, "one2one": one2one}  # one2manyとone2oneを返す

        y = self._inference(one2one)  # 推論を実行
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)  # 後処理を実行
        return y if self.export else (y, {"one2many": x, "one2one": one2one})  # 結果を返す

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # 複数レベルのフィーチャマップに基づいて、予測されたバウンディングボックスとクラス確率をデコードします。
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops。TF FlexSplitV opsを回避
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]  # 分割
        else:  # それ以外の場合
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 特徴を分割

        if self.export and self.format in {"tflite", "edgetpu"}:  # 推論をエクスポートする場合
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)  # バウンディングボックスとクラスを連結

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        # Detect()バイアスを初期化します。警告：ストライドの可用性が必要です。
        m = self  # self.model[-1]  # Detect() module。検出モジュール
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from。fromを反復処理
            a[-1].bias.data[:] = 1.0  # box。ボックス
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)。cls
        if self.end2end:  # エンドツーエンドの場合
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from。fromを反復処理
                a[-1].bias.data[:] = 1.0  # box。ボックス
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)。cls

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        # バウンディングボックスをデコードします。
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)  # バウンディングボックスをデコード

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        # YOLOモデルの予測を後処理します。
        #YOLO の推論結果 (raw predictions) を処理して、最も信頼性の高い検出結果を取得するための後処理 (Post-Processing) を行う関数
        # 引数：
        #     preds (torch.Tensor): 形状(batch_size、num_anchors、4 + nc)の生の予測。最後の次元の形式は[x、y、w、h、class_probs]です。
        #     max_det (int): 画像あたりの最大検出数。
        #     nc (int, optional): クラス数。デフォルト：80。
        #
        # 戻り値：
        #     (torch.Tensor): 形状(batch_size、min(max_det、num_anchors)、6)で、最後の次元形式が[x、y、w、h、max_class_prob、class_index]の処理された予測。
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)。形状を取得
        boxes, scores = preds.split([4, nc], dim=-1)  # 特徴を分割
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)  # インデックスを取得
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))  # ボックスを収集
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))  # スコアを収集
        scores, index = scores.flatten(1).topk(min(max_det, anchors))  # スコアとインデックスを取得
        i = torch.arange(batch_size)[..., None]  # batch indices。バッチインデックス
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)  # 結果を返す
    


class v10Detect(Detect):
    # https://arxiv.org/pdf/2405.14458からのv10検出ヘッド。
    #
    # 引数：
    #     nc (int): クラス数。
    #     ch (tuple): チャネルサイズのタプル。
    #
    # 属性：
    #     max_det (int): 検出の最大数。
    #
    # メソッド：
    #     __init__(self, nc=80, ch=()): v10Detectオブジェクトを初期化します。
    #     forward(self, x): v10Detectモジュールの順方向パスを実行します。
    #     bias_init(self): Detectモジュールのバイアスを初期化します。

    end2end = True  # エンドツーエンド

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        # 指定されたクラス数と入力チャネルでv10Detectオブジェクトを初期化します。
        super().__init__(nc, ch)  # 親クラスを初期化
        c3 = max(ch[0], min(self.nc, 100))  # channels。チャンネル
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )  # 畳み込みレイヤー
        self.one2one_cv3 = copy.deepcopy(self.cv3)  # cv3をコピー
    
    def fuse(self):
        """Removes the one2many head."""
        self.cv2 = self.cv3 = nn.ModuleList([nn.Identity()] * self.nl)


class Detectv2(nn.Module):
    """
    Detectv2は2種類の入力層からそれぞれ別の検出ヘッドを構築します。
    
    入力:
      nc: クラス数
      ch: リストまたはタプルで2要素
          ch[0]: head_a 用の入力チャネルリスト (例: [P3, P4, P5] のチャネル数)
          ch[1]: head_b 用の入力チャネルリスト (例: [P5] のチャネル数)
    """
    end2end = True
    dynamic = False  # force grid reconstruction。グリッド再構築を強制
    export = False  # export mode。エクスポートモード
    max_det = 300  # max_det
    shape = None  # 形状
    anchors = torch.empty(0)  # init。初期化
    strides = torch.empty(0)  # init。初期化
    legacy = False  # backward compatibility for v3/v5/v8/v9 models。v3 / v5 / v8 / v9モデルとの下位互換性

    def __init__(self, nc=80, ch=()):
        super().__init__()
        if not isinstance(ch, list) or len(ch) != 2:
            raise ValueError("ch must be a list of [head_a_channels, head_b_channels]")
        
        self.nc = nc  # number of classes。クラス数
        self.nl = len(ch[0])  # number of detection layers。検出レイヤー数
        self.ch=ch[1]

        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)。DFLチャンネル
        self.no = nc + self.reg_max * 4  # number of outputs per anchor。アンカーあたりの出力数
        self.stride = torch.zeros(self.nl)  # strides computed during build。ビルド中に計算されたストライド
        c2, c3 = max((16, self.ch[0] // 4, self.reg_max * 4)), max(self.ch[0], min(self.nc, 100))  # channels。チャンネル
        "ch[0]とch[1]のチャネル数を一致させるためのconv層"
        self.align_conv = nn.ModuleList(
           nn.Conv2d(input_ch, out_ch, kernel_size=1, stride=1)
           for input_ch, out_ch in zip(ch[0],ch[1])
        )
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in self.ch
        )  # 畳み込みレイヤー
        
        
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in self.ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in self.ch
            )
        )  # 畳み込みレイヤー
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  # DFLを設定  

        self.one2one_cv2 = copy.deepcopy(self.cv2)  # cv2をコピー
        self.one2one_cv3 = copy.deepcopy(self.cv3)  # cv3をコピー  
        
        self.feature_maps = {
            'pre_pan': [],   # 4,6,10の出力 (PAN前)
            'post_pan': [],  # 23,24,25の出力 (1x1 Conv後)
        }  
        
        self.skip_nms = False
        
        self._is_predict = False  # 推論モードフラグの初期化

        
    @property
    def is_predict(self):
        """推論モードかどうかを返す"""
        return self._is_predict

    @is_predict.setter
    def is_predict(self, value):
        """推論モードを設定する"""
        self._is_predict = value

    def forward(self, x):
        """
        入力 x はリストまたはタプルで、以下の形式を期待します:
          x = [head_a_features, head_b_features]
        各 head_x_features は、例: [feat1, feat2, feat3] (各 feat は [B, C, H, W] のテンソル)。
        訓練時は各ヘッドの中間出力を返し、推論時は後処理した検出結果を返します。
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("Input must be a list of [head_a_features, head_b_features]")
    
        pre_pan_feats, post_pan_feats = x
        
        pre_pan_feats_aligned = [
            self.align_conv[i](pre_pan_feats[i]) for i in range(self.nl)
        ]
        
        # self.feature_maps['pre_pan'] = list(pre_pan_feats_aligned)
        # self.feature_maps['post_pan'] = list(post_pan_feats)
        
        one2many = [
            torch.cat((self.cv2[i](post_pan_feats[i]), self.cv3[i](post_pan_feats[i])), 1) for i in range(self.nl)
        ]
        

        #if self.is_predict == False:
        one2one = [
                    torch.cat((self.one2one_cv2[i](pre_pan_feats_aligned[i]), self.one2one_cv3[i](pre_pan_feats_aligned[i])), 1) for i in range(self.nl)
                ]  # one2oneを計算
        # else:
        #     one2one = None
        
        # 特徴を連結
        
        if self.training:  # Training path。トレーニングパスの場合
            return {"one2many": one2many, "one2one": one2one}  # one2manyとone2oneを返す
        

        y = self._inference(one2many)  # 推論を実行
        if self.skip_nms is True:
            y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)  # 後処理を実行
        return y if self.export else (y, {"one2many": one2many, "one2one": one2one})  # 結果を返す

    
    
    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # 複数レベルのフィーチャマップに基づいて、予測されたバウンディングボックスとクラス確率をデコードします。
        # Inference path
        shape = x[0].shape  # BCHW。形状を取得
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 特徴を連結
        if self.dynamic or self.shape != shape:  # 動的または形状が異なる場合
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # アンカーとストライドを生成
            self.shape = shape  # 形状を設定

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops。TF FlexSplitV opsを回避
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]  # 分割
        else:  # それ以外の場合
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 特徴を分割

        if self.export and self.format in {"tflite", "edgetpu"}:  # 推論をエクスポートする場合
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]  # グリッド高
            grid_w = shape[3]  # グリッド幅
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)  # グリッドサイズ
            norm = self.strides / (self.stride[0] * grid_size)  # 正規化
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])  # バウンディングボックスをデコード
        else:  # それ以外の場合
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # バウンディングボックスをデコード

        return torch.cat((dbox, cls.sigmoid()), 1)  # バウンディングボックスとクラスを連結

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        # Detect()バイアスを初期化します。警告：ストライドの可用性が必要です。
        m = self  # self.model[-1]  # Detect() module。検出モジュール
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from。fromを反復処理
            a[-1].bias.data[:] = 1.0  # box。ボックス
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)。cls
        for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from。fromを反復処理
            a[-1].bias.data[:] = 1.0  # box。ボックス
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)。cls

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        # バウンディングボックスをデコードします。
        return dist2bbox(bboxes, anchors, xywh=not self.skip_nms, dim=1)  # バウンディングボックスをデコード

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        # YOLOモデルの予測を後処理します。
        #YOLO の推論結果 (raw predictions) を処理して、最も信頼性の高い検出結果を取得するための後処理 (Post-Processing) を行う関数
        # 引数：
        #     preds (torch.Tensor): 形状(batch_size、num_anchors、4 + nc)の生の予測。最後の次元の形式は[x、y、w、h、class_probs]です。
        #     max_det (int): 画像あたりの最大検出数。
        #     nc (int, optional): クラス数。デフォルト：80。
        #
        # 戻り値：
        #     (torch.Tensor): 形状(batch_size、min(max_det、num_anchors)、6)で、最後の次元形式が[x、y、w、h、max_class_prob、class_index]の処理された予測。
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)。形状を取得
        boxes, scores = preds.split([4, nc], dim=-1)  # 特徴を分割
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)  # インデックスを取得
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))  # ボックスを収集
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))  # スコアを収集
        scores, index = scores.flatten(1).topk(min(max_det, anchors))  # スコアとインデックスを取得
        i = torch.arange(batch_size)[..., None]  # batch indices。バッチインデックス
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)  # 結果を返す




