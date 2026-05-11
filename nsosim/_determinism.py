"""Determinism utility for NSM fitting and decode pipelines.

The NSM fitting / decode path has random sources (latent init, randperm
subsampling, adaptive mesh sampling, cudnn kernel selection). ``set_global_seed``
pins them so two runs with the same seed and inputs produce reproducible outputs.

**Caveats — fit path is not bit-exact on CUDA.**

1. ``torch.nn.functional.grid_sample``'s 2D backward kernel on CUDA has no
   deterministic implementation upstream. NSM's triplanar decoder uses
   grid_sample for feature lookups, so Adam/LBFGS over the latent inherits
   the non-determinism. Decode (forward only) IS bit-exact.

2. **Cross-fit CUDA state** — running multiple NSM fits in the same process
   (e.g. femur → tibia → patella) propagates CUDA workspace and cuBLAS
   layout state. The first fit drifts only from (1) above; subsequent fits
   drift more because their CUDA state depends on prior fits. Subprocess
   isolation per fit (one process per bone) eliminates this. See
   ``kneepipeline/steps/run_nsm.py::_fit_nsm_subprocess``.

3. **Ordering matters when fitting in-process.** ``model.cuda()`` consumes
   CUDA random state during weight transfer. Callers that load a model
   inside their own function must seed AFTER ``load_model`` / ``model.cuda()``.
   ``fit_nsm`` in ``utils.py`` does this.

Together: with order-fix + cudnn flags + use_deterministic_algorithms, a
single bone fit reproduces to ~4-digit agreement on the latent (per
kneepipeline). Multi-bone fits in one process drift more without subprocess
isolation.

Call ``set_global_seed(seed)`` at the entry point of any public API
function that produces meshes (fit, decode, build). It is idempotent.
"""

import logging
import os

logger = logging.getLogger(__name__)

# CUBLAS workspace config is required for deterministic cuBLAS GEMMs when
# torch.use_deterministic_algorithms is on. Set at import time so it's in
# place before any caller imports torch / initializes the CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_global_seed(seed: int = 0) -> None:
    """Pin all random sources for reproducible NSM output.

    Sets PyTorch (CPU + CUDA), NumPy, Python ``random``, cudnn flags, and
    enables ``torch.use_deterministic_algorithms`` in warn-only mode.
    Idempotent — safe to call multiple times.

    ``warn_only=True`` means non-deterministic ops (notably grid_sample
    backward on CUDA) emit a warning instead of raising. We want every
    op that has a deterministic kernel to use it, and a clear warning for
    the ones that don't, without taking down callers.

    Args:
        seed: Seed value to use for all RNGs. Defaults to 0.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.debug("set_global_seed(%d)", seed)
