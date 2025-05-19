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
    
class CrossKDLoss(nn.Module):
    def __init__(self, temperature_cls=3.0, temperature_dfl=3.0):
        super().__init__()
        self.temperature_cls = temperature_cls
        self.temperature_dfl = temperature_dfl
        

    def forward(self, teacher_preds, teacher_head, student_features, fg_mask, reg_max, nc):
        device = student_features[0].device
        total_cls = torch.tensor(0., device=device)
        total_dfl = torch.tensor(0., device=device)
        total_fg = torch.tensor(0., device=device)

        reg4 = reg_max * 4

                # deep copyを使わず、with torch.no_gradブロックで教師ヘッドの勾配計算を防ぐ
        with torch.no_grad():
            # 教師ヘッドの現在のモードを保存
            prev_training = teacher_head.training
            teacher_head.eval()  # 評価モードに設定
            
            try:
                # 学生の特徴を教師のヘッドに通す（勾配計算なし）
                student_cross_preds = [
                    torch.cat((
                        teacher_head.cv2[i](student_features[i]), 
                        teacher_head.cv3[i](student_features[i])
                    ), 1) 
                    for i in range(len(student_features))
                ]
            finally:
                # 教師ヘッドのモードを元に戻す
                teacher_head.train(prev_training)

        # 以下は同じ
        for i, (t_feat, s_feat) in enumerate(zip(teacher_preds, student_cross_preds)):
            _, _, H, W = t_feat.shape
            
            # クラス予測の蒸留（CrossKD）
            t_cls = t_feat[:, reg4:, ...].permute(0,2,3,1).reshape(-1, nc).detach()
            s_cls = s_feat[:, reg4:, ...].permute(0,2,3,1).reshape(-1, nc)
            
            P = F.softmax(t_cls / (self.temperature_cls * 5), dim=1)
            log_Q = F.log_softmax(s_cls / self.temperature_cls, dim=1)
            kld_cls = F.kl_div(log_Q, P, reduction="none").mean(dim=1) * (self.temperature_cls**2)

            # DFL予測の蒸留（CrossKD）
            t_dfl = t_feat[:, :reg4, ...].permute(0,2,3,1).reshape(-1, 4, reg_max).detach()
            s_dfl = s_feat[:, :reg4, ...].permute(0,2,3,1).reshape(-1, 4, reg_max)
            
            P_dfl = F.softmax(t_dfl / (self.temperature_dfl * 5), dim=2)
            log_Q_dfl = F.log_softmax(s_dfl / self.temperature_dfl, dim=2)
            kld_dfl = F.kl_div(log_Q_dfl, P_dfl, reduction="none").mean(dim=(1,2)) * (self.temperature_dfl**2)

            flat_mask = fg_mask[:, i*H*W:(i+1)*H*W].reshape(-1)
            if flat_mask.any():
                total_cls += kld_cls[flat_mask].sum()
                total_dfl += kld_dfl[flat_mask].sum()
                total_fg += flat_mask.sum()

        total_fg = total_fg.clamp(min=1)
        loss = (total_cls + total_dfl) / total_fg

        return loss
    
class CrossKDLoss(nn.Module):
    def __init__(self, temperature_cls=3.0, temperature_dfl=3.0):
        super().__init__()
        self.temperature_cls = temperature_cls
        self.temperature_dfl = temperature_dfl
        

    def forward(self, teacher_preds, teacher_head, student_features, fg_mask, reg_max, nc):
        device = student_features[0].device
        total_cls = torch.tensor(0., device=device)
        total_dfl = torch.tensor(0., device=device)
        total_fg = torch.tensor(0., device=device)

        reg4 = reg_max * 4

                # deep copyを使わず、with torch.no_gradブロックで教師ヘッドの勾配計算を防ぐ
        with torch.no_grad():
            # 教師ヘッドの現在のモードを保存
            prev_training = teacher_head.training
            teacher_head.eval()  # 評価モードに設定
            
            try:
                # 学生の特徴を教師のヘッドに通す（勾配計算なし）
                student_cross_preds = [
                    torch.cat((
                        teacher_head.cv2[i](student_features[i]), 
                        teacher_head.cv3[i](student_features[i])
                    ), 1) 
                    for i in range(len(student_features))
                ]
            finally:
                # 教師ヘッドのモードを元に戻す
                teacher_head.train(prev_training)

        # 以下は同じ
        for i, (t_feat, s_feat) in enumerate(zip(teacher_preds, student_cross_preds)):
            _, _, H, W = t_feat.shape
            
            # クラス予測の蒸留（CrossKD）
            t_cls = t_feat[:, reg4:, ...].permute(0,2,3,1).reshape(-1, nc).detach()
            s_cls = s_feat[:, reg4:, ...].permute(0,2,3,1).reshape(-1, nc)
            
            P = F.softmax(t_cls / (self.temperature_cls * 3), dim=1)
            log_Q = F.log_softmax(s_cls / self.temperature_cls, dim=1)
            kld_cls = F.kl_div(log_Q, P, reduction="none").mean(dim=1) * (self.temperature_cls**2)

            # DFL予測の蒸留（CrossKD）
            t_dfl = t_feat[:, :reg4, ...].permute(0,2,3,1).reshape(-1, 4, reg_max).detach()
            s_dfl = s_feat[:, :reg4, ...].permute(0,2,3,1).reshape(-1, 4, reg_max)
            
            P_dfl = F.softmax(t_dfl / self.temperature_dfl, dim=2)
            log_Q_dfl = F.log_softmax(s_dfl / self.temperature_dfl, dim=2)
            kld_dfl = F.kl_div(log_Q_dfl, P_dfl, reduction="none").mean(dim=(1,2)) * (self.temperature_dfl**2)

            flat_mask = fg_mask[:, i*H*W:(i+1)*H*W].reshape(-1)
            if flat_mask.any():
                total_cls += kld_cls[flat_mask].sum()
                total_dfl += kld_dfl[flat_mask].sum()
                total_fg += flat_mask.sum()

        total_fg = total_fg.clamp(min=1)
        loss = (total_cls + total_dfl) / total_fg

        return loss