import torch.nn as nn
from models.resnet import ResNET18
from models.efficientnet import EfficientNet5C
from models.mobilevit import mobile_vit_xx_small
from models.pinn import RamanTransformerClassifier
from models.dynattn import DynamicPathCrossAttention
import torch
import torch.nn.functional as F

# efficientnet_rp_foll_seed9.pth    pinn_rp_foll_seed1.pth
# efficientnet_rp_plasma_seed1.pth  pinn_rp_pla_seed5.pth
# mobilevit_rp_foll_seed1.pth       resnet18_rp_follseed1.pth
# mobilevit_rp_plasma_seed1.pth     resnet18_rp_plaseed9.pth

class REMClassifier(nn.Module):
    def __init__(self, in_channels=5, num_classes=2,):
        super().__init__()
        # pinn_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/pinn_rp_foll_seed1.pth'
        # res_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/resnet18_rp_follseed1.pth'
        # eff_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/efficientnet_rp_foll_seed9.pth'
        # mb_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/mobilevit_rp_foll_seed1.pth'

        pinn_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/pinn_rp_pla_seed5.pth'
        res_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/resnet18_rp_plaseed9.pth'
        eff_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/efficientnet_rp_plasma_seed1.pth'
        mb_weight_path = '/root/data/zhj/PCOSclassfication/checkpoints/mobilevit_rp_plasma_seed1.pth'

        self.res_model = ResNET18(in_channels=in_channels, num_classes=2)
        self.eff_model = EfficientNet5C(in_channels=in_channels, num_classes=2)
        self.mbvit_model = mobile_vit_xx_small(in_channels=in_channels, num_classes=2)
        self.pinn = RamanTransformerClassifier(in_channels=in_channels, num_classes=2)

        self.pinn.load_state_dict(torch.load(pinn_weight_path, map_location='cpu'))
        self.res_model.load_state_dict(torch.load(res_weight_path, map_location='cpu'))
        self.eff_model.load_state_dict(torch.load(eff_weight_path, map_location='cpu'))
        self.mbvit_model.load_state_dict(torch.load(mb_weight_path, map_location='cpu'))

        self.res_model.model.fc = nn.Identity()
        self.eff_model.model._fc = nn.Identity()
        self.mbvit_model.classifier = nn.Identity()
        self.pinn.fc_out = nn.Identity()

        self.project_res = nn.Linear(512, 256)
        self.project_eff = nn.Linear(1280, 256)
        self.project_mbvit = nn.Linear(320, 256)
        self.project_pinn = nn.Linear(1024*128, 256)

        self.dynattn = DynamicPathCrossAttention(embed_dim=256, num_paths=3, top_k=2)

        self.classifier = nn.Linear(256, num_classes)


    def forward(self, images,signals):
        # 特征提取
        with torch.no_grad():
            feat_pinn = self.pinn(signals)  # [B, 1024*128]
            feat_res = self.res_model(images)  # [B, 512]
            feat_eff = self.eff_model(images)  # [B, 1280]
            feat_mbvit = self.mbvit_model(images)  # [B, 960]

        # 投影到相同维度
        feat_pinn_proj = self.project_pinn(feat_pinn).unsqueeze(1)  # [B, 1, 256]
        feat_res_proj = self.project_res(feat_res).unsqueeze(1)     # [B, 1, 256]
        feat_eff_proj = self.project_eff(feat_eff).unsqueeze(1)     # [B, 1, 256]
        feat_mbvit = F.adaptive_avg_pool2d(feat_mbvit, 1)  # [B, 320, 1, 1]
        feat_mbvit = feat_mbvit.view(feat_mbvit.size(0), -1)  # [B, 320]
        feat_mbvit_proj = self.project_mbvit(feat_mbvit).unsqueeze(1)  # [B, 1, 256]

        # 模态特征融合 [B, 1, 256]
        fusion_feat = self.dynattn(feat_pinn_proj, [feat_res_proj, feat_eff_proj, feat_mbvit_proj])

        # 输出层
        fusion_feat = fusion_feat.squeeze(1)  # [B, 256]
        outputs = self.classifier(fusion_feat)
        return outputs

