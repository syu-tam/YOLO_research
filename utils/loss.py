# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import OKS_SIGMA
from utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from utils.tal import  TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from nn.modules import Detectv2
    
    
class VarifocalLoss(nn.Module):
    # ZhangらによるVarifocal損失。
    # https://arxiv.org/abs/2008.13367。

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        # VarifocalLossクラスを初期化します。
        super().__init__()  # 親クラスを初期化

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        # varfocal損失を計算します。
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label  # 重みを計算
        with autocast(enabled=False):  # 自動混合精度を無効化
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )  # 損失を計算
        return loss  # 損失を返す


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""
    # focal損失を既存のloss_fcn（）の周りにラップします。つまり、criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)です。

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        # パラメータなしのFocalLossクラスのイニシャライザ。
        super().__init__()  # 親クラスを初期化

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        # オブジェクト検出/分類タスクの混同行列を計算および更新します。
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")  # 損失を計算
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits。ロジットからの確率
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)  # p_tを計算
        modulating_factor = (1.0 - p_t) ** gamma  # 変調係数を計算
        loss *= modulating_factor  # 損失を調整
        if alpha > 0:  # alphaが0より大きい場合
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)  # alpha係数を計算
            loss *= alpha_factor  # 損失を調整
        return loss.mean(1).sum()  # 損失を返す


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""
    # トレーニング中にDFL損失を計算するための基準クラス。

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        # DFLモジュールを初期化します。
        super().__init__()  # 親クラスを初期化
        self.reg_max = reg_max  # reg_maxを設定

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        # 左右のDFL損失の合計を返します。
        #
        # 一般化焦点損失で提案されている分散焦点損失（DFL）
        # https://ieeexplore.ieee.org/document/9792391
        target = target.clamp_(0, self.reg_max - 1 - 0.01)  # targetをクランプ
        tl = target.long()  # target left。ターゲット左
        tr = tl + 1  # target right。ターゲット右
        wl = tr - target  # weight left。重み左
        wr = 1 - wl  # weight right。重み右
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)  # 損失を計算して返す


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

class PANFeatureLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.feature_weights = [1.5, 1.0, 0.5]

    def forward(self, pre_pan_features, post_pan_features):
        pan_loss = 0.0
        with autocast(enabled=False):  # 自動混合精度 (AMP) を適用しない
            for i, (pre, post) in enumerate(zip(pre_pan_features, post_pan_features)):
                loss = self.mse(pre, post)*self.feature_weights[i]
                pan_loss += loss
        return pan_loss
    

class E2EDetectLoss:
    """Criterion class for computing training losses."""
    # トレーニング損失を計算するための基準クラス。

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        # 提供されたモデルを使用して、1対多および1対1の検出損失でE2EDetectLossを初期化します。
        self.one2many = v8DetectionLoss(model, tal_topk=10, use_pan_loss = True)  # 1対多の損失を初期化
        self.one2one = v8DetectionLoss(model, tal_topk=10, use_pan_loss = False)  # 1対1の損失を初期化
        
        # 蒸留用のパラメータ
        self.temperature = 1.0
        self.alpha_cls = 0.5  # クラス分類の蒸留重み
        self.alpha_dfl = 0.5  # DFL分布の蒸留重み

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # バッチサイズを掛けた、ボックス、クラス、およびdflの損失の合計を計算します。
        preds = preds[1] if isinstance(preds, tuple) else preds  # 予測を取得
        one2many = preds["one2many"]  # one2manyを取得
        loss_one2many = self.one2many(one2many, batch)  # 1対多の損失を計算
        one2one = preds["one2one"]  # one2oneを取得
        loss_one2one = self.one2one(one2one, batch) # 1対1の損失を計算 
        
        # # 蒸留損失の計算 ここから------------------------------------
        distill_loss = 0.25 * self._compute_distillation_loss(one2many, one2one) 
        
        # loss_one2many[1] の4番目に distill_loss を加える
        stats_tensor = loss_one2many[1].clone().detach()  # 元のテンソルを複製（in-place操作を避ける）
        stats_tensor[3] += distill_loss.clone().detach()  # 4番目の要素に加算
        total_loss = (loss_one2many[0] + loss_one2one[0] + distill_loss)
        return total_loss, stats_tensor + loss_one2one[1]
        # 蒸留損失の計算 ここまで------------------------------------
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1] # 損失を返す
    
    def _compute_distillation_loss(self, teacher_preds, student_preds):
        """クラス分類とDFL分布の蒸留損失を計算"""
        kl_loss = torch.tensor(0., device=self.one2many.device)
        teacher_preds = torch.cat([xi.view(teacher_preds[0].shape[0], self.one2many.no, -1) for xi in teacher_preds ], 2)
        student_preds = torch.cat([xi.view(student_preds[0].shape[0], self.one2one.no, -1) for xi in student_preds], 2)    
        for t_feat, s_feat in zip(teacher_preds, student_preds):
            
            # 予測を分離（DFLとクラス分類）
            t_dfl, t_cls = t_feat.split((self.one2many.reg_max * 4, self.one2many.nc), 0)
            s_dfl, s_cls = s_feat.split((self.one2many.reg_max * 4, self.one2many.nc), 0)
            
            # クラス分類の蒸留
            t_cls_prob = (t_cls / self.temperature).softmax(dim=1)
            s_cls_prob = (s_cls / self.temperature).log_softmax(dim=1)
            cls_kld = F.kl_div(
                s_cls_prob, 
                t_cls_prob, 
                reduction='batchmean'
            ) * (self.temperature ** 2)

            # DFL分布の蒸留
            t_dfl = t_dfl.view(-1, 4, self.one2many.reg_max)
            s_dfl = s_dfl.view(-1, 4, self.one2many.reg_max)
            
            t_dfl_prob = (t_dfl / self.temperature).softmax(dim=2)
            s_dfl_prob = (s_dfl / self.temperature).log_softmax(dim=2)
            dfl_kld = F.kl_div(
                s_dfl_prob, 
                t_dfl_prob, 
                reduction='batchmean'
            ) * (self.temperature ** 2)

            # 重み付け合算
            kl_loss += self.alpha_cls * cls_kld + self.alpha_dfl * dfl_kld

        return kl_loss
    
    

class v8DetectionLoss:
    """Criterion class for computing training losses."""
    # トレーニング損失を計算するための基準クラス。

    def __init__(self, model, tal_topk=10, use_pan_loss=True):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        # モデルを使用してv8DetectionLossを初期化し、モデル関連のプロパティとBCE損失関数を定義します。
        device = next(model.parameters()).device  # get model device。モデルデバイスを取得
        h = model.args  # hyperparameters。ハイパーパラメータ
        
        m = model.model[-1]  # Detect() module。最後のレイヤーを取得
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # 損失関数
        self.hyp = h  # ハイパーパラメータを設定
        self.stride = m.stride  # model strides。モデルストライド
        self.nc = m.nc  # number of classes。クラス数
        self.no = m.nc + m.reg_max * 4  # Number of outputs per anchor。アンカーあたりの出力数
        self.reg_max = m.reg_max  # reg max。reg_maxを設定
        self.device = device  # デバイスを設定

        self.use_dfl = m.reg_max > 1  # DFLの使用
        self.use_pan_loss = use_pan_loss
    
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)  # タスクアライメントアサインを初期化
        self.bbox_loss = BboxLoss(m.reg_max).to(device)  # バウンディングボックス損失を初期化
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)  # プロジェクトを初期化
        
        # self.pan_loss = PANFeatureLoss().to(device)  # PAN特徴マップ損失を追加
        # self.attn_loss = SquaredSumAttentionTransferLoss().to(device)
        # self.channel_dist = ChannelWiseDistillation(temperature=2.0).to(device)
        
        if isinstance(model.model[-1], Detectv2):
            if tal_topk == 1:
                model.model[-1].skip_nms = True  # Detectv2にtal_topkを設定


    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # ターゲットカウントを前処理し、入力バッチサイズと一致させてテンソルを出力します。
        nl, ne = targets.shape  # ターゲットの形状を取得
        if nl == 0:  # ターゲットがない場合
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)  # ゼロテンソルを初期化
        else:  # ターゲットがある場合
            i = targets[:, 0]  # image index。画像インデックス
            _, counts = i.unique(return_counts=True)  # 一意のカウント
            counts = counts.to(dtype=torch.int32)  # 型を設定
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)  # ゼロテンソルを初期化
            for j in range(batch_size):  # バッチサイズを反復処理
                matches = i == j  # 一致するもの
                n = matches.sum()  # 合計
                if n:  # 一致するものがある場合
                    out[j, :n] = targets[matches, 1:]  # ターゲットをコピー
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))  # スケールを適用
        return out  # 結果を返す

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        # アンカーポイントと分布から予測されたオブジェクトのバウンディングボックス座標をデコードします。
        if self.use_dfl:  # DFLを使用する場合
            b, a, c = pred_dist.shape  # batch, anchors, channels。バッチ、アンカー、チャネル
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))  # デコード
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)  # 距離からバウンディングボックスを計算

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # バッチサイズを掛けた、ボックス、クラス、およびdflの損失の合計を計算します。
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl。損失を初期化
        feats = preds[1] if isinstance(preds, tuple) else preds  # フィーチャを取得
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # 予測を連結して分割

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # 順列を適用
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # 順列を適用

        dtype = pred_scores.dtype  # データ型を取得
        batch_size = pred_scores.shape[0]  # バッチサイズを取得
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)。画像サイズを計算
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # アンカーを作成

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)  # ターゲットを連結
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # ターゲットを前処理
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy。ラベルとバウンディングボックスを取得
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # マスクを生成

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)。バウンディングボックスをデコード
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )  # アサインを実行

        target_scores_sum = max(target_scores.sum(), 1)  # ターゲットスコアの合計

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way。VFL法
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE。BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        # # PAN特徴マップ損失の計算を追加
        # if hasattr(self.model.model[-1], 'feature_maps') and self.use_pan_loss == True:
        #     pre_features = self.model.model[-1].feature_maps['pre_pan']
        #     post_features = self.model.model[-1].feature_maps['post_pan']
        #     if pre_features and post_features:
        #         #loss[3] = self.channel_dist(pre_features, post_features)
        #         #loss[3] = self.attn_loss(pre_features, post_features)
        #         loss[3] = self.pan_loss(pre_features, post_features)
        

        loss[0] *= self.hyp.box  # box gain。ボックスゲイン
        loss[1] *= self.hyp.cls  # cls gain。clsゲイン
        loss[2] *= self.hyp.dfl  # dfl gain。dflゲイン
        #loss[3] *= 0 # PAN損失の重みを適用
        
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)。損失を返す

class ChannelWiseDistillation(nn.Module):
    def __init__(self, temperature=1.0):
        """
        Channel-wise Distillation Loss の実装
        - 各チャンネルの特徴マップを確率分布化し、KL ダイバージェンスを計算
        """
        super().__init__()
        self.temperature = temperature  # 温度パラメータ T

    def forward(self, teacher_features, student_features):
        """
        教師 (Teacher) と生徒 (Student) の特徴マップを入力し、スケールごとの KL ダイバージェンスを計算

        Args:
            teacher_features (list of torch.Tensor): 教師の特徴マップ [(B, C1, H1, W1), (B, C2, H2, W2), (B, C3, H3, W3)]
            student_features (list of torch.Tensor): 生徒の特徴マップ [(B, C1', H1', W1'), (B, C2', H2', W2'), (B, C3', H3', W3')]

        Returns:
            torch.Tensor: スケールごとの平均 KL ダイバージェンス損失
        """
        total_loss = 0.0
        num_scales = len(teacher_features)  # スケールの数（通常3）

        for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
            # 各スケールの特徴マップのサイズを取得
            B, C, H, W = t_feat.shape

            # Softmax による確率分布化（各チャンネルごとに H×W に対して）
            t_prob = F.softmax(t_feat.view(B, C, -1) / self.temperature, dim=-1).view(B, C, H, W)
            s_prob = F.softmax(s_feat.view(B, C, -1) / self.temperature, dim=-1).view(B, C, H, W)

            # KLダイバージェンスの計算（バッチ平均）
            loss = F.kl_div(s_prob.log(), t_prob, reduction='batchmean')

            # 論文の式通りに T^2 を適用
            total_loss += ((self.temperature ** 2) / C) * loss

        return total_loss   # スケールごとの平均損失を返す

class SquaredSumAttentionTransferLoss(nn.Module):
    """
    論文と同様の手法で2乗和のAttention Transfer Lossを計算するクラスです。
    
    各特徴マップ（pre_features, post_features）について、
    チャネル方向の2乗和をとって空間的注意マップを生成し、各サンプルごとに
    ベクトル化・l2正規化します。その後、教師と生徒の正規化済み注意マップの
    L2距離（ノルム）を計算して合計します。
    
    Attributes:
        loss_weight (float): 損失に乗じる重み（β相当、デフォルトは1.0）
    """
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, teacher_features, student_features):
        """
        教師と生徒の特徴マップリストからAttention Transfer Lossを計算します。
        
        Args:
            teacher_features (list of torch.Tensor): 教師側の特徴マップリスト。各テンソルは形状 (B, C, H, W)。
            student_features (list of torch.Tensor): 生徒側の特徴マップリスト。各テンソルは形状 (B, C, H, W)。
        
        Returns:
            torch.Tensor: 損失値（スカラー）
        """
        total_loss = 0.0
        with autocast(enabled=True):
            # 各層/スケールごとに注意マップを計算し、L2距離を算出
            for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
                # F^2_sumによる注意マップの計算: チャネル方向の2乗和 → (B, H, W)
                # その後、フラット化して (B, H*W) に、さらにl2正規化
                t_att = self._compute_attention(t_feat)
                s_att = self._compute_attention(s_feat)
                # 各サンプルごとにL2ノルム（p=2）の差を計算し、バッチ平均
                layer_loss = (s_att - t_att).norm(p=2, dim=1).mean()
                total_loss += layer_loss
        return self.loss_weight * total_loss

    def _compute_attention(self, feat):
        """
        特徴マップから注意マップを計算します。
        
        入力 feat は形状 (B, C, H, W) とし、チャネル方向の2乗和をとって (B, H, W) の
        注意マップを生成、その後 (B, H*W) にフラット化し、各サンプルごとにl2正規化します。
        
        Args:
            feat (torch.Tensor): 特徴マップ、形状 (B, C, H, W)
        
        Returns:
            torch.Tensor: l2正規化された注意マップ、形状 (B, H*W)
        """
        # チャネルごとの2乗和（F^2_sum）
        att = (feat ** 2).sum(dim=1)  # (B, H, W)
        # 空間次元をフラット化
        att = att.view(att.size(0), -1)  # (B, H*W)
        # 各サンプルごとにl2正規化
        att = F.normalize(att, p=2, dim=1)
        return att