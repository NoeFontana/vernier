"""Re-export of ``mask_to_boundary`` from the parity_boundary vendor.

The upstream ``boundary-iou-api`` repo (``bowenc0221``) ships a single
``boundary_iou.utils.boundary_utils`` module that both the
``coco_instance_api`` and the ``coco_panoptic_api`` subpackages import
from. The instance-side vendor lives at
``tests/python/parity_boundary/oracle/boundary_iou_api/boundary_iou/utils/``
at the same pinned SHA (``37d25586a677b043ed585f10e5c42d4e80176ea9``).

Rather than duplicate ``boundary_utils.py`` under this oracle subtree
(and risk drift between two copies), we re-export ``mask_to_boundary``
from the parity_boundary vendor. The panoptic conftest places that
oracle root on ``sys.path`` so the import below resolves to the same
verbatim upstream copy used by the instance/boundary parity suite.

This file is the **only** non-verbatim file in this oracle subtree. The
upstream ``coco_panoptic_api/evaluation.py`` uses
``from ..utils import mask_to_boundary``; resolving that relative
import to a single source of truth is what this shim exists for.
"""

from boundary_iou.utils.boundary_utils import mask_to_boundary  # noqa: F401
