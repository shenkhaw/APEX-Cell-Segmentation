# APEX: Adaptive Pathology Extraction

A generalizable Mixture of Experts (MoE) framework for universal cell segmentation in spatial transcriptomics.

## Overview
Standard cell segmentation tools (like Cellpose) fail on dense pathology (e.g., amyloid plaques, tumors, fibrosis) because they mistake the pathology for giant cells, resulting in massive data loss. 

APEX solves this by using a shared Attention U-Net encoder to extract universal tissue geometry, which routes to three specialized decoders (fuzzy, sharp, streaky) to accurately mask out pathology and rescue the underlying cells.

## Architecture
- **Stage 1:** Universal morphology pre-training (IHC, H&E, PSR) with aggressive color/intensity augmentation.
- **Stage 2:** Target domain fine-tuning with Macenko stain normalization and Boundary Loss.
- **Routing:** Sparse Mixture of Experts (MoE) with Oracle Routing for stable training.

*(Code coming soon)*
