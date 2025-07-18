import os
import torch
from torch import nn
from functools import partial
import nibabel as nib
from src.utils import safe_get_source

import rff
import logging

logging.basicConfig(level=logging.INFO)


def safe_get_source(sources: list):
    assert len(set(sources)) == 1
    return sources[0]


class MaskRegistered(nn.Module):
    def __init__(self):
        super().__init__()
        for subj in range(1, 9):
            path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                f"data/roi/subj0{subj}/nsdgeneral.nii.gz",
            )
            visual_mask = torch.as_tensor(nib.load(path).get_fdata()).long() > 0
            self.register_buffer(f"visual_mask_subject_{subj}", visual_mask)

    def get_mask(self, subject):
        if isinstance(subject, list):
            subject = safe_get_source(subject)
        mask = getattr(self, f"visual_mask_{subject}")
        return mask

    def get_masks(self, subjects):
        return [getattr(self, f"visual_mask_{subject}") for subject in subjects]


class NeuroscienceInformedAttentionLayer(nn.Module):
    def __init__(
        self,
        size,
        rank=512,
    ):
        super().__init__()
        self.size = size
        self.rank = rank

        self.query_embeddings = nn.Parameter(torch.randn(1, size, rank))
        nn.init.xavier_uniform_(self.query_embeddings)

    def forward(self, values, keys=None, context=None):
        """
        query - (2, B, S, H)
        values - (B, L)
        keys - (B, L, H)
        """
        query = self.query_embeddings.unsqueeze(1).expand(
            -1, values.size(0), -1, -1
        )  # (2, B, S, H)
        unnormalized_weights = query @ keys.transpose(1, 2).unsqueeze(0)  # (2, B, S, L)

        weight = unnormalized_weights[0]  # (B, S, L)
        weight = weight.softmax(-1)
        out = (values.unsqueeze(-2) @ weight.transpose(1, 2)).squeeze(1)
        return out


class NeuroscienceInformedAttention(MaskRegistered):
    def __init__(
        self,
        n_fmri_tokens=128,
        token_dim=1024,
        rank=128,
        pe_method="gauss",
        hidden_dim=1024,
        act_first=False,
        n_mlp_layers=4,
        norm_type="ln",
    ):
        super().__init__()
        self.n_fmri_tokens = n_fmri_tokens
        self.pe_method = pe_method
        out_dim = n_fmri_tokens * token_dim
        norm_func = (
            partial(nn.BatchNorm1d, num_features=hidden_dim)
            if norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=hidden_dim)
        )
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == "bn" else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.neuro_informed_attn = NeuroscienceInformedAttentionLayer(
            size=hidden_dim, rank=rank
        )
        assert pe_method in ["embed", "sin", "coord", "gauss", "none"]
        if self.pe_method == "sin":
            self.coords_encoding = PositionalEncoding(128, max_len=100)
            self.coords_transform = nn.Sequential(
                nn.Linear(128 * 3, 128 * 3),
                nn.GELU(),
                nn.Linear(128 * 3, 128 * 3),
                nn.GELU(),
                nn.Linear(128 * 3, rank),
            )
        elif self.pe_method == "coord":
            self.coords_transform = nn.Sequential(
                nn.Linear(3, rank),
                nn.GELU(),
                nn.Linear(rank, rank),
                nn.GELU(),
                nn.Linear(rank, rank),
            )
        elif self.pe_method == "gauss":
            self.coords_encoding = rff.layers.GaussianEncoding(
                sigma=5.0, input_size=3, encoded_size=192
            )
            self.coords_transform = nn.Sequential(
                nn.Linear(192 * 2, rank),
                nn.GELU(),
                nn.Linear(rank, rank),
                nn.GELU(),
                nn.Linear(rank, rank),
            )
        else:
            raise ValueError("Unsupported pe method.")

        self.neuro_informed_attn_post = nn.Sequential(
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.n_mlp_layers = n_mlp_layers
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    *[item() for item in act_and_norm],
                    nn.Dropout(0.15),
                )
                for _ in range(n_mlp_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

        rois = {
            name: {
                f"subject_{subj}": torch.as_tensor(
                    nib.load(f"data/roi/subj0{subj}/{name}.nii.gz")
                    .get_fdata()
                    .astype("int")
                )
                for subj in range(1, 9)
            }
            for name in [
                "HCP_MMP1",
                "Kastner2015",
                # 'corticalsulc',
                # 'streams',
                "prf-visualrois",
                # 'prf-eccrois',
                "floc-faces",
                # 'floc-words',
                "floc-places",
                # 'floc-bodies',
                # 'thalamus',
                # 'MTL',
            ]
        }
        n_regions = {name: 0 for name in rois.keys()}
        for name in rois.keys():
            for subject_i, roi_of_subject in rois[name].items():
                self.register_buffer(f"roi_{name}_{subject_i}", roi_of_subject)
                n_regions[name] = max(n_regions[name], roi_of_subject.max() + 1)
        roi_embed_dim = 32
        self.roi_embeds = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(n_regions[name], roi_embed_dim))
                for name, roi in rois.items()
            }
        )
        for key, value in self.roi_embeds.items():
            nn.init.xavier_uniform_(value)
        if self.pe_method == "none":
            self.region_feature_project = nn.Linear(roi_embed_dim * len(rois), rank)
        else:
            self.region_feature_project = nn.Linear(
                roi_embed_dim * len(rois) + rank, rank
            )
        self.region_feature_project = nn.Linear(roi_embed_dim * len(rois) + rank, rank)

    def forward(self, voxels, **kwargs):
        B = voxels.size(0)
        subject = safe_get_source(kwargs["subject"])
        mask = self.get_mask(kwargs["subject"])

        if voxels.dim() > 2:
            voxels = voxels.masked_select(mask.unsqueeze(0)).view(B, -1)
        if self.pe_method == "sin":
            coords1, coords2, coords3 = map(
                lambda x: self.coords_encoding(
                    x.unsqueeze(0).expand(B, -1).contiguous()
                ),
                torch.where(mask),
            )
            coords = self.coords_transform(
                torch.cat([coords1, coords2, coords3], dim=-1)
            )
        elif self.pe_method == "coord":
            coords = torch.stack(torch.where(mask))
            coords = coords.float() + 0.5
            coords = (
                coords.transpose(0, 1)
                .unsqueeze(0)
                .expand(voxels.size(0), -1, -1)
                .contiguous()
            )
            coords = self.coords_transform(coords / 50.0)
        elif self.pe_method == "gauss":
            coords = torch.stack(torch.where(mask))
            coords = (
                coords.transpose(0, 1)
                .unsqueeze(0)
                .expand(voxels.size(0), -1, -1)
                .contiguous()
            )
            encoding = self.coords_encoding(coords)
            coords = self.coords_transform(encoding)
        else:
            raise

        roi_embeds = []
        for roi_name, roi_embeddings in self.roi_embeds.items():
            roi = getattr(self, f"roi_{roi_name}_{subject}")
            region = roi[mask]
            roi_embeds.append(roi_embeddings[region])
        roi_embeds = torch.cat(roi_embeds, dim=-1).unsqueeze(0).expand(B, -1, -1)

        keys = self.region_feature_project(torch.cat([coords, roi_embeds], dim=-1))

        x = self.neuro_informed_attn(voxels, keys)

        x = self.neuro_informed_attn_post(x)
        residual = x
        for res_block in range(self.n_mlp_layers):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.head(x)
        x = x.reshape(voxels.size(0), self.n_fmri_tokens, -1)
        return x
