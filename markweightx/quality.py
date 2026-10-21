"""Quality metrics the pipeline reports next to detection."""
from __future__ import annotations


def fid(images_a: str, images_b: str, *, cuda: bool = False) -> float:
    """Frechet Inception Distance between two directories of PNG images (torch-fidelity, as last night's runs).
    Compare sets of equal size: FID depends on the count."""
    try:
        import torch_fidelity
    except ImportError:
        raise ImportError("fid needs torch-fidelity: pip install torch-fidelity") from None
    m = torch_fidelity.calculate_metrics(input1=images_a, input2=images_b, cuda=cuda, fid=True, verbose=False)
    return float(m["frechet_inception_distance"])


def rel_l2(pred, truth) -> float:
    """Relative L2 error ||pred - truth|| / ||truth||, averaged over the batch (GeoPT's metric)."""
    p, t = pred.float().flatten(1), truth.float().flatten(1)
    return float(((p - t).norm(dim=1) / t.norm(dim=1)).mean())
