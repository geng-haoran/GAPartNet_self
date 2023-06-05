from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Segmentation:
    batch_size: int

    sem_preds: torch.Tensor
    sem_labels: Optional[torch.Tensor] = None

