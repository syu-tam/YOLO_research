# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from utils.metrics import OKS_SIGMA
from utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from utils.tal import  TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from nn.modules import Detectv2
import math
from torchvision.ops import box_iou  
from typing import List
    
    
class VarifocalLoss(nn.Module):
    # ZhangらによるVarifocal損失。
    # https://arxiv.org/abs/2008.13367。

    def __init__(self, gamma=2.0, alpha=0.75):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score, gt_score, label):
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )  # 損失を計算
        return loss  # 損失を返す


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Args:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float | list): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma=1.5, alpha=0.25):
        """Initialize FocalLoss class with no parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred, label):
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


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
    """
    Feature distillation loss between pre- and post-PAN feature maps.
    正例のみを対象に、target_scoresで重みづけしたチャネル平均MSEを集計し、
    全正例スコアで正規化して返す。
    """
    def __init__(self):
        super().__init__()
        # per-pixel の MSE を計算するため reduction='none'
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pre_pan_feats, post_pan_feats,target_scores, fg_mask):
        device = target_scores.device

        target_scores = target_scores.sum(-1) # B , 8400
        target_scores_sum = max(target_scores.sum(), 1)  # ターゲットスコアの合計

        # 各スケールごとの空間画素数 (H*W) で分割するためのサイズリスト
        spatial_sizes = [feat.size(2) * feat.size(3) for feat in pre_pan_feats] # B , H1*W1

    
        # target_scores と fg_mask をスケールごとに分割
        score_splits   = target_scores.split(spatial_sizes, dim=1)# B , H1*W1
        mask_splits    = fg_mask.split(spatial_sizes, dim=1)# B , H1*W1

        # 加重 MSE の累積変数
        weighted_mse_sum = torch.tensor(0.0, device=device)

        # 各スケールについて計算
        for pre_feat, post_feat, scores_i, mask_i in zip(
                pre_pan_feats, post_pan_feats, score_splits, mask_splits):

            # チャネル方向を平均して per-pixel MSE マップを作成 → [B, H, W]
            per_pixel_mse_map = self.mse(pre_feat, post_feat.detach()).mean(dim=1)

           
            # Flatten: [B*H*W]
            flat_mse    = per_pixel_mse_map.view(-1)
            flat_scores = scores_i.reshape(-1)
            flat_mask   = mask_i.reshape(-1).bool()

            # 正例ピクセルのみに絞り、スコアで重みをかけて合計
            weighted_mse_sum += (flat_mse * flat_scores)[flat_mask].sum()
        
        feature_loss = weighted_mse_sum / target_scores_sum # 正規化して返却

        # 正規化して返却
        return feature_loss


class DistillationLoss(nn.Module):
    def __init__(self, 
                 temperature_cls=3.0, 
                 temperature_dfl=3.0, 
                 confidence_threshold=0.5,
                 iou_threshold=0.5):
        super().__init__()
        self.temperature_cls = temperature_cls
        self.temperature_dfl = temperature_dfl
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
    def forward(self, teacher_preds, student_preds, fg_mask, reg_max, nc):
        device = teacher_preds[0].device
        total_cls = torch.tensor(0., device=device)
        total_dfl = torch.tensor(0., device=device)
        total_fg  = torch.tensor(0., device=device)

        with autocast(enabled=False):
            teacher_preds = [t.detach().float() for t in teacher_preds]
            student_preds = [s.float() for s in student_preds]

            T     = self.temperature_cls
            T_dfl = self.temperature_dfl
            reg4  = reg_max * 4

            # スケールごとに損失を集計
            for i, (t_feat, s_feat) in enumerate(zip(teacher_preds, student_preds)):
                _, _, H, W = t_feat.shape
                # === クラス蒸留 KLD ===
                t_cls = (t_feat[:, reg4:, ...].permute(0,2,3,1).reshape(-1, nc))
                s_cls = (s_feat[:, reg4:, ...].permute(0,2,3,1).reshape(-1, nc))
                
                # 教師モデルの確信度を計算
                with torch.no_grad():
                    t_prob = torch.sigmoid(t_cls / T)
                    t_conf = torch.sigmoid(t_cls).max(dim=1)[0]
            
                # バイナリクロスエントロピーベースの蒸留
                kld_cls = F.binary_cross_entropy_with_logits(
                    s_cls / T, t_prob, reduction="none"
                ).mean(dim=1) * (T**2)

                # === DFL蒸留 KLD ===
                t_dfl = (t_feat[:, :reg4, ...].permute(0,2,3,1).reshape(-1, 4, reg_max))
                s_dfl = (s_feat[:, :reg4, ...].permute(0,2,3,1).reshape(-1, 4, reg_max))

                with torch.no_grad():
                    t_prob_dfl = (t_dfl / T_dfl).softmax(dim=2)
                
                kld_dfl = -(t_prob_dfl * F.log_softmax(s_dfl / T_dfl, dim=2)).mean(dim=(1,2)) * (T_dfl**2)

                # === 正例マスクと確信度マスクの組み合わせ ===
                flat_mask = fg_mask[:, i*H*W:(i+1)*H*W].reshape(-1)
                conf_mask = t_conf >= self.confidence_threshold
                combined_mask = flat_mask & conf_mask

                if combined_mask.any():
                    total_cls += kld_cls[combined_mask].sum()
                    total_dfl += kld_dfl[combined_mask].sum()
                    total_fg  += combined_mask.sum()

            # まとめて正規化
            total_fg = total_fg.clamp(min=1)
            loss_per_fg = (total_cls + total_dfl) / total_fg

        return loss_per_fg


class E2EDetectLoss:
    """Criterion class for computing training losses."""
    # トレーニング損失を計算するための基準クラス。

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        # 提供されたモデルを使用して、1対多および1対1の検出損失でE2EDetectLossを初期化します。
        self.main_loss = v8DetectionLoss(model, tal_topk=10)  # 1対多の損失を初期化
        self.aux_loss = v8DetectionLoss(model, tal_topk=10)  # 1対1の損失を初期化

        device = next(model.parameters()).device

        self.distill_losses = {
            'pan': PANFeatureLoss().to(device),
            'cross_kd': DistillationLoss(
                temperature_cls=3.0,
                temperature_dfl=3.0,
                confidence_threshold=0.5  # 確信度のしきい値を追加
            ).to(device)
        }
        
        self.distill_weights = {
            'pan': 0.0,
            'cross_kd': 0.05
        }

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # バッチサイズを掛けた、ボックス、クラス、およびdflの損失の合計を計算します。
        preds = preds[1] if isinstance(preds, tuple) else preds  # 予測を取得

        # 検出損失の計算
        loss_main, loss_aux = self.compute_detection_losses(
            preds['main_head'], 
            preds['aux_head'], 
            batch
        )

                # 蒸留損失の計算
        distill_losses = self.compute_distillation_losses(
            preds, 
            self.main_loss.last_fg_mask
        )
    
        # 統計情報の更新
        stats = loss_main[1].clone().detach()
        stats[3] += distill_losses['pan'].detach()
        stats[4] += distill_losses['cross_kd'].detach()
        
        # 最終的な損失の計算
        total_loss = (
            loss_main[0] + 
            0.5 * loss_aux[0] + 
            sum(distill_losses.values())
        )
        
        return total_loss, stats + 0.5 * loss_aux[1]
    
    def compute_detection_losses(self, main_head, aux_head, batch):
        """検出損失の計算"""
        loss_main = self.main_loss(main_head, batch)
        loss_aux = self.aux_loss(aux_head, batch)
        return loss_main, loss_aux

    def compute_distillation_losses(self, features, fg_mask):
        """蒸留損失の計算"""
        losses = {}
        
        # PAN特徴量の蒸留損失
        losses['pan'] = self.distill_weights['pan'] * self.distill_losses['pan'](
            features['pre_pan'],
            features['post_pan'],
            self.main_loss.target_scores,
            fg_mask
        )
        
        # CrossKD損失
        losses['cross_kd'] = self.distill_weights['cross_kd'] * self.distill_losses['cross_kd'](
            teacher_preds=features['main_head'],
            student_preds=features['aux_head'],
            fg_mask=fg_mask,
            reg_max=self.main_loss.reg_max,
            nc=self.main_loss.nc
        )
        
        return losses


class v8DetectionLoss:
    """Criterion class for computing training losses."""
    # トレーニング損失を計算するための基準クラス。

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
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
    
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)  # タスクアライメントアサインを初期化
        self.bbox_loss = BboxLoss(m.reg_max).to(device)  # バウンディングボックス損失を初期化
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)  # プロジェクトを初期化

        self.model = model
        
        if isinstance(model.model[-1], Detectv2) and tal_topk == 11:
            model.model[-1].skip_nms = True  # Detectv2にtal_topkを設定
        # マスク保存用
        self.last_fg_mask = None

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
                if n := matches.sum():  # 一致するものがある場合
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
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl。損失を初期化
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

        self.target_scores = target_scores # ターゲットスコアの合計を保存

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way。VFL法
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE。BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        
        self.last_fg_mask = fg_mask  # (B, A)

        loss[0] *= self.hyp.box  # box gain。ボックスゲイン
        loss[1] *= self.hyp.cls  # cls gain。clsゲイン
        loss[2] *= self.hyp.dfl  # dfl gain。dflゲイン
        
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)。損失を返す

