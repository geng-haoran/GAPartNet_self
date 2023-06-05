import lightning.pytorch as lp
from typing import Optional, Dict, Tuple, List
import functools
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
import torch.nn.functional as F
from network.losses import focal_loss
from network.losses import dice_loss

from structure.point_cloud import PointCloudBatch, PointCloud
from structure.segmentation import Segmentation

class GAPartNet(lp.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_part_classes: int,
        backbone_type: str = "SparseUNet",
        backbone_cfg: Optional[Dict] = None,
        learning_rate: float = 1e-3,
        # semantic segmentation
        ignore_sem_label: int = -100,
        use_sem_focal_loss: bool = True,
        use_sem_dice_loss: bool = True,  
    ):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        
        self.in_channels = in_channels
        self.num_part_classes = num_part_classes
        self.backbone_type = backbone_type
        self.backbone_cfg = backbone_cfg
        self.learning_rate = learning_rate
        self.ignore_sem_label = ignore_sem_label
        self.use_sem_focal_loss = use_sem_focal_loss
        self.use_sem_dice_loss = use_sem_dice_loss
        
        ## network
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        # backbone
        if self.backbone_type == "SparseUNet":
            from .backbone import SparseUNet
            channels = self.backbone_cfg["channels"]
            block_repeat = self.backbone_cfg["block_repeat"]
            self.backbone = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        else:
            raise NotImplementedError(f"backbone type {self.backbone_type} not implemented")
        # semantic segmentation head
        self.sem_seg_head = nn.Linear(channels[0], self.num_part_classes)
        
    def forward_backbone(
        self,
        pc_batch: PointCloudBatch,
    ):
        if self.backbone_type == "SparseUNet":
            voxel_tensor = pc_batch.voxel_tensor
            pc_voxel_id = pc_batch.pc_voxel_id
            voxel_features = self.backbone(voxel_tensor)
            pc_feature = voxel_features.features[pc_voxel_id]

        return pc_feature
    
    def forward_sem_seg(
        self,
        pc_feature: torch.Tensor,
    ) -> Tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor]:
        sem_logits = self.sem_seg_head(pc_feature)

        return sem_logits


    def loss_sem_seg(
        self,
        sem_logits: torch.Tensor,
        sem_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_sem_focal_loss:
            loss = focal_loss(
                sem_logits, sem_labels,
                alpha=None,
                gamma=2.0,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )
        else:
            loss = F.cross_entropy(
                sem_logits, sem_labels,
                weight=None,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )

        if self.use_sem_dice_loss:
            loss += dice_loss(
                sem_logits[:, :, None, None], sem_labels[:, None, None],
            )

        return loss


    def _training_or_validation_step(
        self,
        point_clouds: List[PointCloud],
        batch_idx: int,
        running_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(point_clouds)
        data_batch = PointCloud.collate(point_clouds)
        points = data_batch.points
        sem_labels = data_batch.sem_labels
        pc_ids = data_batch.pc_ids
        
        pt_xyz = points[:, :3]
        # cls_labels.to(pt_xyz.device)

        pc_feature = self.forward_backbone(pc_batch=data_batch)

        # semantic segmentation
        sem_logits = self.forward_sem_seg(pc_feature)
        
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        if sem_labels is not None:
            loss_sem_seg = self.loss_sem_seg(sem_logits, sem_labels)
        else:
            loss_sem_seg = 0.
        # import pdb; pdb.set_trace()

        sem_seg = Segmentation(batch_size=batch_size,sem_preds=sem_preds,sem_labels=sem_labels,)
        


        # total loss
        loss = loss_sem_seg
        all_accu = (sem_preds == sem_labels).sum().float() / (sem_labels.shape[0])
        
        # if sem_labels is not None:
        #     instance_mask = sem_labels > 0
        #     pixel_acc = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
        # else:
        #     pixel_acc = 0.0

        prefix = running_mode
        self.log(
            f"{prefix}/total_loss", 
            loss, 
            batch_size=batch_size,
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            f"{prefix}/loss_sem_seg",
            loss_sem_seg,
            batch_size=batch_size,
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"{prefix}/all_accu",
            all_accu * 100,
            batch_size=batch_size,
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        return pc_ids, sem_seg, all_accu, loss

    def training_step(self, point_clouds: List[PointCloud], batch_idx: int):
        _, _, _, loss = self._training_or_validation_step(
            point_clouds, batch_idx, "train"
        )

        return loss

    def validation_step(self, point_clouds: List[PointCloud], batch_idx: int, dataloader_idx: int = 0):
        split = ["val", "intra", "inter"]
        pc_ids, sem_seg, all_accu, _ = self._training_or_validation_step(
            point_clouds, batch_idx, split[dataloader_idx]
        )
        if dataloader_idx > len(self.validation_step_outputs) - 1:
            self.validation_step_outputs.append([])
        self.validation_step_outputs[dataloader_idx].append((pc_ids, sem_seg, all_accu))
        return pc_ids, sem_seg, all_accu

    def on_validation_epoch_end(self):
        
        splits = ["val", "intra", "inter"]
        sem_accu = {}
        for i_, validation_step_outputs in enumerate(self.validation_step_outputs):
            split = splits[i_]
            batch_size = sum(x[1].batch_size for x in validation_step_outputs)
            all_accu = sum(x[2] for x in validation_step_outputs) / len(validation_step_outputs)
            # torch.save(validation_step_outputs, "wandb/predictions_gap.pth")
            del validation_step_outputs
            sem_accu[split] = all_accu
            self.log(f"{split}/all_accu", 
                    all_accu, 
                    batch_size=batch_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.log(f"moniter_metric/sem_accu", 
                sem_accu["intra"] + sem_accu["inter"], 
                batch_size=batch_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.validation_step_outputs.clear() 

    def test_step(self, point_clouds: List[PointCloud], batch_idx: int, dataloader_idx: int = 0):
        split = ["val", "intra", "inter"]
        pc_ids, sem_seg, proposals, _ = self._training_or_validation_step(
            point_clouds, batch_idx, split[dataloader_idx]
        )

        if proposals is not None:
            proposals = filter_invalid_proposals(
                proposals,
                score_threshold=self.val_score_threshold,
                min_num_points_per_proposal=self.val_min_num_points_per_proposal
            )
            proposals = apply_nms(proposals, self.val_nms_iou_threshold)

        if proposals != None:
            proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
            proposals.valid_mask = None
            proposals.pt_xyz = None
            proposals.sem_preds = None
            proposals.npcs_preds = None
            proposals.sem_labels = None
            proposals.npcs_valid_mask = None
            proposals.gt_npcs = None

        return pc_ids, sem_seg, proposals

    def test_epoch_end(self, validation_step_outputs_list):
        splits = ["val", "intra", "inter"]
        for i_, validation_step_outputs in enumerate(validation_step_outputs_list):
            split = splits[i_]
        
            batch_size = sum(x[1].batch_size for x in validation_step_outputs)

            proposals = [x[2] for x in validation_step_outputs] # if x[2] != None 
            
            del validation_step_outputs

            if proposals[0] is not None:

                aps = compute_ap(proposals, self.num_classes, self.val_ap_iou_threshold)

                for class_idx in range(1, self.num_classes):
                    partname = PART_ID2NAME[class_idx]
                    self.log(
                        f"{split}/AP@50_{partname}",
                        aps[class_idx - 1] * 100,
                        batch_size=batch_size,
                        on_epoch=True, prog_bar=True, logger=True
                        
                    )
                self.log(
                    f"{split}/AP@50", 
                    np.mean(aps) * 100, 
                    batch_size=batch_size,
                    on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
