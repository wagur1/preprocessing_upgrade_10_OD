"""Preprocessing framework for video machine vision under compression.

Implements the framework from "A Preprocessing Framework for Video Machine
Vision under Compression" (Zhao et al., arXiv:2512.15331), with the paper's
hand-crafted differentiable *virtual codec* replaced by a CompressAI learned
codec (bmshj2018-factorized), whose factorized-prior entropy model is exactly
the Balle et al. model the paper cites for its rate estimate.
"""

__version__ = "0.1.0"
