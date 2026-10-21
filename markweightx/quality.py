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


def accuracy(outputs, golds) -> float:
    """Share of outputs that give their gold answer. Multiple choice (every gold a letter A-J, MedXpertQA): the last
    "answer is (X)" in the output, as exp/med/evaluate.py; otherwise the output contains the gold answer (lower case,
    no punctuation or articles), as the agent demo."""
    import re
    if golds and all(isinstance(g, str) and re.fullmatch(r"[A-J]", g) for g in golds):
        pick = lambda o: (re.findall(r"answer is\s*\(?\s*([A-J])\b", str(o), re.I) or [None])[-1]
        return sum((pick(o) or "").upper() == g for o, g in zip(outputs, golds)) / len(golds)
    from .models.agent import norm
    return sum(bool(o) and norm(str(g)) in norm(str(o)) for o, g in zip(outputs, golds)) / max(len(golds), 1)


def ppl(model, tok, prompts, texts) -> float:
    """Perplexity of texts under the unwatermarked base (the release's adapter switched off)."""
    from .sampling import perplexity
    return perplexity(model, tok, prompts, texts, adapter=False)
