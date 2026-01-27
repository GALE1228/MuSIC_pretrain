import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResidualBlock1D, ResidualBlock2D
from .se import SEBlock
from .Conv2d import Conv2d

# Dimensionality reduction and upsampling
class FeatureProjector(nn.Module):
    def __init__(self, seq_dim=1280, stru_dim=2, target_dim=64):
        super(FeatureProjector, self).__init__()
        # embedding
        self.seq_fc = nn.Linear(seq_dim, target_dim)
        # structure
        self.stru_fc = nn.Linear(stru_dim, target_dim)
        # final dimension
        self.total_dim = target_dim * 2

    def forward(self, x):
        seq_part = x[:, :, :1280]  # (B, 200, 1280)
        stru_part = x[:, :, 1280:]  # (B, 200, 2)
        seq_proj = self.seq_fc(seq_part)   # (B, 200, 64)
        stru_proj = self.stru_fc(stru_part) # (B, 200, 64)
        out = torch.cat([seq_proj, stru_proj], dim=-1)  # (B, 200, 128)
        return out

# RNA encoder model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.projector = FeatureProjector(seq_dim=1280, stru_dim=2, target_dim=64)
        self.n_features = self.projector.total_dim
        
        h_p, h_k = 1, 3
        base_channel = 32
        self.conv    = Conv2d(in_channels = 1, out_channels = base_channel, kernel_size=(h_k, h_k), bn = True, same_padding=True)
        self.se      = SEBlock(base_channel)
        self.res2d   = ResidualBlock2D(base_channel, kernel_size=(h_k, h_k), padding=(h_p,h_p)) 
        self.res1d   = ResidualBlock1D(base_channel*4)
        self.avgpool = nn.AvgPool2d((self.n_features,1))
        self.gpool   = nn.AdaptiveAvgPool1d(1)
        # self.fc      = nn.Linear(base_channel*4*8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rna_input):
        """[forward]
        
        Args:
            input ([tensor],N,C,W,H): input features
        """
        rna_input = self.projector(rna_input) # (B, 200, 128)
        
        rna_input = rna_input.transpose(1, 2)  # (B, 128, 200)
        x = self.conv(rna_input) # (B, 32, 128, 200)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x) # (B, 32, 1, 1)
        x = self.res2d(x*z) # (B, 32 * 4, 128 ,200)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x) # (B, 32 * 4, 1 ,200)
        x = x.view(x.shape[0], x.shape[1], x.shape[3])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1]) # (B,1024)
        # x = self.fc(x)
        return x

# RNA-Protein Interaction module
class RNAProteinInteraction(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super(RNAProteinInteraction, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, r_repr, p_repr):
        # r_repr: [B, Lr, D], p_repr: [B, Lp, D]
        attn_out, attn_weights = self.cross_attention(query=r_repr, key=p_repr, value=p_repr) # [B, 1, D]
        out = self.norm(r_repr + attn_out)       # residual + LayerNorm
        out = self.norm(out + self.ffn(out))     # FFN + residual
        return out

# MLP module
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc2 = nn.Linear(dim // 4, dim // 16)
        self.fc3 = nn.Linear(dim // 16, 1)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, RPR):
        """[forward]
        
        Args:
            RPR : RNAProteinRepresentation features
        """
        
        x = self.fc1(RPR)
        x = F.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.fc3(x)
        return x
    
# MuSIC model
class MuSIC(nn.Module):
    def __init__(self, dim, n_heads=8):
        super(MuSIC, self).__init__()
        self.rna_encoder = CNN()
        self.interaction = RNAProteinInteraction(dim=dim, n_heads=n_heads)
        self.mlp = MLP(dim)

    def forward(self, rna, rbp_repr):
        """
        Args:
            rna: [B, 1, n_features, seq_len]
            rbp_repr: [B, Lp, D]  (protein embedding)
        Returns:
            prediction: [B, 1]
        """
        device = rna.device
        rbp_repr = rbp_repr.to(device)
        
        B = rna.shape[0]
        if rbp_repr.dim() == 2:  # [L, 1024]
            rbp_repr = rbp_repr.unsqueeze(0).repeat(B, 1, 1)
        
        r_repr = self.rna_encoder(rna)       # [B, 1024]
        r_repr = r_repr.unsqueeze(1)             # [B, 1, D]
        joint_repr = self.interaction(r_repr, rbp_repr)  # [B, 1, D]
        joint_repr = joint_repr.squeeze(1)               # [B, D]
        pred = self.mlp(joint_repr)
        return pred