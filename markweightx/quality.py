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
