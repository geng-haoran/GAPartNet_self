import lightning.pytorch as lp
from typing import Optional, Dict, Tuple, List
import functools
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
import torch.nn.functional as F

from epic_ops.reduce import segmented_maxpool
from epic_ops.iou import batch_instance_seg_iou

from network.losses import focal_loss, dice_loss, pixel_accuracy, mean_iou
from network.grouping_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)
from structure.point_cloud import PointCloudBatch, PointCloud
from structure.segmentation import Segmentation
from structure.instances import Instances

from misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID
from misc.visu import visualize_gapartnet

class GAPartNet(lp.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_part_classes: int,
        backbone_type: str = "SparseUNet",
        backbone_cfg: Dict = {},
        learning_rate: float = 1e-3,
        # semantic segmentation
        ignore_sem_label: int = -100,
        use_sem_focal_loss: bool = True,
        use_sem_dice_loss: bool = True,
        # instance segmentation
        instance_seg_cfg: Dict = {},
        # training
        training_schedule: List = [],
        # validation
        val_score_threshold: float = 0.09,
        val_min_num_points_per_proposal: int = 3,
        val_nms_iou_threshold: float = 0.3,
        val_ap_iou_threshold: float = 0.5,
        # testing
        visualize_cfg: Dict = {},
        
        debug: bool = True,
        ckpt: str = "", # type: ignore
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
        self.visualize_cfg = visualize_cfg
        self.start_scorenet, self.start_npcs = training_schedule
        self.start_clustering = min(self.start_scorenet, self.start_npcs)
        self.val_nms_iou_threshold = val_nms_iou_threshold
        self.val_ap_iou_threshold = val_ap_iou_threshold
        self.val_score_threshold = val_score_threshold
        self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        
        self.ball_query_radius = instance_seg_cfg["ball_query_radius"]
        self.max_num_points_per_query = instance_seg_cfg["max_num_points_per_query"]
        self.min_num_points_per_proposal = instance_seg_cfg["min_num_points_per_proposal"]
        self.max_num_points_per_query_shift = instance_seg_cfg["max_num_points_per_query_shift"]
        self.score_fullscale = instance_seg_cfg["score_fullscale"]
        self.score_scale = instance_seg_cfg["score_scale"]
        
        
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
        # offset prediction
        self.offset_head = nn.Sequential(
            nn.Linear(channels[0], 2*channels[0]),
            norm_fn(2*channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(2*channels[0], 3),
        )
        
        self.score_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.score_head = nn.Linear(channels[0], self.num_part_classes - 1)

        
        if ckpt != "":
            print("Loading pretrained model from:", ckpt)
            state_dict = torch.load(
                ckpt, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
        
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
    ) -> torch.Tensor:
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

    def forward_offset(
        self,
        pc_feature: torch.Tensor,
    ) -> torch.Tensor:
        offset = self.offset_head(pc_feature)

        return offset
    
    def loss_offset(
        self,
        offsets: torch.Tensor,
        gt_offsets: torch.Tensor,
        sem_labels: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_instance_mask = (sem_labels > 0) & (instance_labels >= 0)

        pt_diff = offsets - gt_offsets
        pt_dist = torch.sum(pt_diff.abs(), dim=-1)
        loss_offset_dist = pt_dist[valid_instance_mask].mean()

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=-1)
        gt_offsets = gt_offsets / (gt_offsets_norm[:, None] + 1e-8)

        offsets_norm = torch.norm(offsets, p=2, dim=-1)
        offsets = offsets / (offsets_norm[:, None] + 1e-8)

        dir_diff = -(gt_offsets * offsets).sum(-1)
        loss_offset_dir = dir_diff[valid_instance_mask].mean()

        return loss_offset_dist, loss_offset_dir

    def proposal_clustering_and_revoxelize(
        self,
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        pt_features: torch.Tensor,
        sem_preds: torch.Tensor,
        offset_preds: torch.Tensor,
        instance_labels: Optional[torch.Tensor],
    ):
        device = self.device
        
        if instance_labels is not None:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = sem_preds > 0
        
        pt_xyz = pt_xyz[valid_mask]
        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        offset_preds = offset_preds[valid_mask]
        if instance_labels is not None:
            instance_labels = instance_labels[valid_mask]
            
        # get batch offsets (csr) from batch indices
        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)
        
        # cluster proposals: dual set
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + offset_preds, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
        )
        
        # combine clusters
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)

        # compact the proposal ids
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # remove small proposals
        valid_proposal_mask = (
            num_points_per_proposal >= self.min_num_points_per_proposal
        )
        # proposal to point
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            import pdb; pdb.set_trace()
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        if instance_labels is not None:
            instance_labels = instance_labels[sorted_indices]

        # re-compact the proposal ids
        proposal_indices = proposal_indices[valid_point_mask]
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            proposal_indices, return_inverse=True, return_counts=True
        )
        num_proposals = num_points_per_proposal.shape[0]

        # get proposal batch offsets
        proposal_offsets = torch.zeros(
            num_proposals + 1, dtype=torch.int32, device=device
        )
        proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

        # voxelization
        voxel_features, voxel_coords, pc_voxel_id = segmented_voxelize(
            pt_xyz, pt_features,
            proposal_offsets, proposal_indices,
            num_points_per_proposal,
            self.score_fullscale, self.score_scale,
        )
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(),
            spatial_shape=[self.score_fullscale] * 3,
            batch_size=num_proposals,
        )
        if not (pc_voxel_id >= 0).all():
            import pdb
            pdb.set_trace()
            


        proposals = Instances(
            valid_mask=valid_mask,
            sorted_indices=sorted_indices,
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            proposal_offsets=proposal_offsets,
            proposal_indices=proposal_indices,
            num_points_per_proposal=num_points_per_proposal,
            sem_preds=sem_preds,
            instance_labels=instance_labels,
        )

        return voxel_tensor, pc_voxel_id, proposals

    def forward_proposal_score(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
        proposals: Instances,
    ):
        proposal_offsets = proposals.proposal_offsets
        proposal_offsets_begin = proposal_offsets[:-1] # type: ignore
        proposal_offsets_end = proposal_offsets[1:] # type: ignore

        score_features = self.score_unet(voxel_tensor)
        score_features = score_features.features[pc_voxel_id]
        pooled_score_features, _ = segmented_maxpool(
            score_features, proposal_offsets_begin, proposal_offsets_end
        )
        score_logits = self.score_head(pooled_score_features)

        return score_logits

    def loss_proposal_score(
        self,
        score_logits: torch.Tensor,
        proposals: Instances,
        num_points_per_instance: torch.Tensor,
    ) -> torch.Tensor:
        ious = batch_instance_seg_iou(
            proposals.proposal_offsets, # type: ignore
            proposals.instance_labels, # type: ignore
            proposals.batch_indices, # type: ignore
            num_points_per_instance,
        )
        proposals.ious = ious
        proposals.num_points_per_instance = num_points_per_instance

        ious_max = ious.max(-1)[0]
        gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

        return F.binary_cross_entropy_with_logits(score_logits, gt_scores)


    def _training_or_validation_step(
        self,
        point_clouds: List[PointCloud],
        batch_idx: int,
        running_mode: str,
    ):
        batch_size = len(point_clouds)
        
        # data batch parsing
        data_batch = PointCloud.collate(point_clouds)
        points = data_batch.points
        sem_labels = data_batch.sem_labels
        pc_ids = data_batch.pc_ids
        instance_regions = data_batch.instance_regions
        instance_labels = data_batch.instance_labels
        batch_indices = data_batch.batch_indices
        instance_sem_labels = data_batch.instance_sem_labels
        num_points_per_instance = data_batch.num_points_per_instance
        
        
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
        
        # accuracy
        all_accu = (sem_preds == sem_labels).sum().float() / (sem_labels.shape[0])
        
        if sem_labels is not None:
            instance_mask = sem_labels > 0
            pixel_accu = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
        else:
            pixel_accu = 0.0
            
        sem_seg = Segmentation(
            batch_size=batch_size,
            sem_preds=sem_preds,
            sem_labels=sem_labels,
            all_accu=all_accu,
            pixel_accu=pixel_accu,)
        
        offsets_preds = self.forward_offset(pc_feature)
        if instance_regions is not None:
            offsets_gt = instance_regions[:, :3] - pt_xyz
            loss_offset_dist, loss_offset_dir = self.loss_offset(
                offsets_preds, offsets_gt, sem_labels, instance_labels, # type: ignore
            )
        else:
            import pdb; pdb.set_trace()
            loss_offset_dist, loss_offset_dir = 0., 0.

        if self.current_epoch >= self.start_clustering:
            voxel_tensor, pc_voxel_id, proposals = self.proposal_clustering_and_revoxelize(
                pt_xyz = pt_xyz,
                batch_indices=batch_indices,
                pt_features=pc_feature,
                sem_preds=sem_preds,
                offset_preds=offsets_preds,
                instance_labels=instance_labels,
            )
            
            if sem_labels is not None and proposals is not None:
                proposals.sem_labels = sem_labels[proposals.valid_mask][
                    proposals.sorted_indices
                ]
            if proposals is not None:
                proposals.instance_sem_labels = instance_sem_labels
                
                
        # clustering and scoring
        if self.current_epoch >= self.start_scorenet and voxel_tensor is not None and proposals is not None: # type: ignore
            score_logits = self.forward_proposal_score(
                voxel_tensor, pc_voxel_id, proposals
            ) # type: ignore
            proposal_offsets_begin = proposals.proposal_offsets[:-1].long() # type: ignore

            if proposals.sem_labels is not None: # type: ignore
                proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long() # type: ignore
            else:
                proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long() # type: ignore
            score_logits = score_logits.gather(
                1, proposal_sem_labels[:, None] - 1
            ).squeeze(1)
            proposals.score_preds = score_logits.detach().sigmoid() # type: ignore
            if num_points_per_instance is not None: # type: ignore
                loss_prop_score = self.loss_proposal_score(
                    score_logits, proposals, num_points_per_instance, # type: ignore
                )
            else:
                import pdb; pdb.set_trace()
                loss_prop_score = 0.0


        # total loss
        loss = loss_sem_seg + loss_offset_dist + loss_offset_dir + loss_prop_score


        prefix = running_mode
        # losses
        self.log(
            f"{prefix}_loss/total_loss", 
            loss, 
            batch_size=batch_size,
            on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(
            f"{prefix}_loss/loss_sem_seg",
            loss_sem_seg,
            batch_size=batch_size,
            on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )
        self.log(
            f"{prefix}_loss/loss_offset_dist",
            loss_offset_dist,
            batch_size=batch_size,
            on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )
        self.log(
            f"{prefix}_loss/loss_offset_dir",
            loss_offset_dir,
            batch_size=batch_size,
            on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )
        self.log(
            f"{prefix}_loss/loss_prop_score",
            loss_prop_score,
            batch_size=batch_size,
            on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )
        
        # evaulation metrics
        self.log(
            f"{prefix}/all_accu",
            all_accu * 100,
            batch_size=batch_size,
            on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )
        self.log(
            f"{prefix}/pixel_accu",
            pixel_accu * 100,
            batch_size=batch_size,
            on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )

        return pc_ids, sem_seg, proposals, loss

    def training_step(self, point_clouds: List[PointCloud], batch_idx: int):
        _, _, proposals, loss = self._training_or_validation_step(
            point_clouds, batch_idx, "train"
        )
        return loss

    def validation_step(self, point_clouds: List[PointCloud], batch_idx: int, dataloader_idx: int = 0):
        split = ["val", "test_intra", "test_inter"]
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
        
        
        if dataloader_idx > len(self.validation_step_outputs) - 1:
            self.validation_step_outputs.append([])
        
        proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
        

        
        
        proposals_ = Instances(
            score_preds=proposals.score_preds, pt_sem_classes=proposals.pt_sem_classes, \
            batch_indices=proposals.batch_indices, instance_sem_labels=proposals.instance_sem_labels, \
            ious=proposals.ious, proposal_offsets=proposals.proposal_offsets, valid_mask= proposals.valid_mask)
        
        
        
        self.validation_step_outputs[dataloader_idx].append((pc_ids, sem_seg, proposals_))
        return pc_ids, sem_seg, proposals_

    def on_validation_epoch_end(self):
        
        
        splits = ["val", "test_intra", "test_inter"]
        all_accus = []
        pixel_accus = []
        mious = []
        mean_ap50 = []
        mAPs = []
        for i_, validation_step_outputs in enumerate(self.validation_step_outputs):
            split = splits[i_]
            pc_ids = [i for x in validation_step_outputs for i in x[0]]
            batch_size = validation_step_outputs[0][1].batch_size
            data_size = sum(x[1].batch_size for x in validation_step_outputs)
            all_accu = sum(x[1].all_accu for x in validation_step_outputs) / len(validation_step_outputs)
            pixel_accu = sum(x[1].pixel_accu for x in validation_step_outputs) / len(validation_step_outputs)
            
            
            # semantic segmentation
            sem_preds = torch.cat(
                [x[1].sem_preds for x in validation_step_outputs], dim=0
            )
            sem_labels = torch.cat(
                [x[1].sem_labels for x in validation_step_outputs], dim=0
            )
            miou = mean_iou(sem_preds, sem_labels, num_classes=self.num_part_classes)
            
            # instance segmentation
            proposals = [x[2] for x in validation_step_outputs if x[2]!= None]
            
            del validation_step_outputs
            
            # semantic segmentation
            all_accus.append(all_accu)
            mious.append(miou)
            pixel_accus.append(pixel_accu)
            
            # instance segmentation
            thes = [0.5 + 0.05 * i for i in range(10)]
            aps = []
            for the in thes:
                ap = compute_ap(proposals, self.num_part_classes, the)
                aps.append(ap)
                if the == 0.5:
                    ap50 = ap
            mAP = np.array(aps).mean()
            mAPs.append(mAP)

            for class_idx in range(1, self.num_part_classes):
                partname = PART_ID2NAME[class_idx]
                self.log(
                    f"{split}/AP@50_{partname}",
                    np.mean(ap50[class_idx - 1]) * 100,
                    batch_size=data_size,
                    on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
                )
                
            
            mean_ap50.append(np.mean(ap50) * 100)
            
            
            self.log(f"{split}/AP@50", 
                    np.mean(ap50) * 100, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{split}/mAP", 
                    mAP * 100, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{split}/all_accu", 
                    all_accu * 100.0, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{split}/pixel_accu", 
                    pixel_accu * 100.0, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{split}/miou",
                     miou * 100.0,
                     batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        self.log("monitor_metrics/mean_all_accu", 
                (all_accus[1]+all_accus[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_pixel_accu", 
                (pixel_accus[1]+pixel_accus[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_imou", 
                (mious[1]+mious[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_AP@50", 
                (mean_ap50[1]+mean_ap50[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_mAP", 
                (mAPs[1]+mAPs[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True
                )



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

        
        if dataloader_idx > len(self.validation_step_outputs) - 1:
            self.validation_step_outputs.append([])
        
        proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
        
        
        # # NMS and filter
        # if proposals is not None:
        #     proposals = filter_invalid_proposals(
        #         proposals,
        #         score_threshold=self.val_score_threshold,
        #         min_num_points_per_proposal=self.val_min_num_points_per_proposal
        #     )
        #     proposals = apply_nms(proposals, self.val_nms_iou_threshold)
        
        
        proposals_ = Instances(
            score_preds=proposals.score_preds, 
            pt_sem_classes=proposals.pt_sem_classes,
            batch_indices=proposals.batch_indices, 
            instance_sem_labels=proposals.instance_sem_labels,
            ious=proposals.ious, 
            proposal_offsets=proposals.proposal_offsets, 
            proposal_indices=proposals.proposal_indices, 
            valid_mask= proposals.valid_mask,
            num_points_per_proposal=proposals.num_points_per_proposal,
            num_points_per_instance=proposals.num_points_per_instance,
            sorted_indices = proposals.sorted_indices,
        )
        
        self.validation_step_outputs[dataloader_idx].append((pc_ids, sem_seg, proposals_))
        return pc_ids, sem_seg, proposals_

    def on_test_epoch_end(self):
        
        splits = ["val", "test_intra", "test_inter"]
        all_accus = []
        pixel_accus = []
        mious = []
        mean_ap50 = []
        mAPs = []
        for i_, validation_step_outputs in enumerate(self.validation_step_outputs):
            split = splits[i_]
            pc_ids = [i for x in validation_step_outputs for i in x[0]]
            batch_size = validation_step_outputs[0][1].batch_size
            data_size = sum(x[1].batch_size for x in validation_step_outputs)
            all_accu = sum(x[1].all_accu for x in validation_step_outputs) / len(validation_step_outputs)
            pixel_accu = sum(x[1].pixel_accu for x in validation_step_outputs) / len(validation_step_outputs)
            
            
            # semantic segmentation
            sem_preds = torch.cat(
                [x[1].sem_preds for x in validation_step_outputs], dim=0
            )
            sem_labels = torch.cat(
                [x[1].sem_labels for x in validation_step_outputs], dim=0
            )
            miou = mean_iou(sem_preds, sem_labels, num_classes=self.num_part_classes)
            
            # instance segmentation
            proposals = [x[2] for x in validation_step_outputs if x[2]!= None]
            
            del validation_step_outputs
            
            # semantic segmentation
            all_accus.append(all_accu)
            mious.append(miou)
            pixel_accus.append(pixel_accu)
            
            # instance segmentation
            thes = [0.5 + 0.05 * i for i in range(10)]
            aps = []
            for the in thes:
                ap = compute_ap(proposals, self.num_part_classes, the)
                aps.append(ap)
                if the == 0.5:
                    ap50 = ap
            mAP = np.array(aps).mean()
            mAPs.append(mAP)

            for class_idx in range(1, self.num_part_classes):
                partname = PART_ID2NAME[class_idx]
                self.log(
                    f"{split}/AP@50_{partname}",
                    np.mean(ap50[class_idx - 1]) * 100,
                    batch_size=data_size,
                    on_epoch=True, prog_bar=False, logger=True, sync_dist=True,
                )
                
            
            mean_ap50.append(np.mean(ap50) * 100)

            
            if self.visualize_cfg["visualize"] == True:
                if self.visualize_cfg["sample_num"] > 0:
                    import random
                    sample_ids = random.sample(range(len(pc_ids)), self.visualize_cfg["sample_num"])
                else:
                    sample_ids = range(len(pc_ids))
                
                
                for sample_id in sample_ids:
                    batch_id = sample_id // batch_size
                    batch_sample_id = sample_id % batch_size
                    proposals_ = proposals[batch_id]
                    mask = proposals_.valid_mask.reshape(-1,20000)[batch_sample_id]
                    
                    if proposals_ is not None:
                        pt_xyz = proposals_.pt_xyz
                        batch_indices = proposals_.batch_indices
                        proposal_offsets = proposals_.proposal_offsets
                        num_points_per_proposal = proposals_.num_points_per_proposal
                        num_proposals = num_points_per_proposal.shape[0]
                        npcs_preds = proposals_.npcs_preds
                        score_preds= proposals_.score_preds
                        mask = proposals_.valid_mask

                        indices = torch.arange(mask.shape[0], dtype=torch.int64,device = sem_preds.device)
                        proposal_indices = indices[proposals_.valid_mask][proposals_.sorted_indices]
                        
                        ins_seg_preds =  torch.ones(mask.shape[0]) * 0
                        for ins_i in range(len(proposal_offsets) - 1):
                            ins_seg_preds[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = ins_i+1
                    
                    sample_sem_pred = sem_preds.reshape(-1,20000)[sample_id]    
                    sample_ins_seg_pred = ins_seg_preds.reshape(-1,20000)[batch_sample_id]
                    # import pdb; pdb.set_trace()
                    
                    visualize_gapartnet(
                        SAVE_ROOT=self.visualize_cfg["SAVE_ROOT"],
                        RAW_IMG_ROOT = self.visualize_cfg["RAW_IMG_ROOT"],
                        GAPARTNET_DATA_ROOT=self.visualize_cfg["GAPARTNET_DATA_ROOT"],
                        save_option=self.visualize_cfg["save_option"],
                        name = pc_ids[sample_id],
                        split = split,
                        sem_preds=sample_sem_pred.cpu().numpy(), # type: ignore
                        ins_preds=sample_ins_seg_pred.cpu().numpy(),
                    )
            
            
            
            self.log(f"{split}/AP@50", 
                    np.mean(ap50) * 100, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{split}/mAP", 
                    mAP * 100, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{split}/all_accu", 
                    all_accu * 100.0, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{split}/pixel_accu", 
                    pixel_accu * 100.0, 
                    batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{split}/miou",
                     miou * 100.0,
                     batch_size=data_size,
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        # need to make sure the order of the splits is correct:
        # the second validation set is intra set and the third set is inter set
        self.log("monitor_metrics/mean_all_accu", 
                (all_accus[1]+all_accus[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_pixel_accu", 
                (pixel_accus[1]+pixel_accus[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_imou", 
                (mious[1]+mious[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_AP@50", 
                (mean_ap50[1]+mean_ap50[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("monitor_metrics/mean_mAP", 
                (mAPs[1]+mAPs[2])/2 * 100.0, 
                batch_size=data_size,
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True
                )


        self.validation_step_outputs.clear() 
    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
