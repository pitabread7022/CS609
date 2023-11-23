Objective:

Imputation and denoising of spatial transcriptomic data

Spatially resolved transcriptomics (SRT) provides gene expression close to, or even superior
to, single-cell resolution while retaining the physical locations of sequencing and often also
providing matched pathology images. However, SRT expression data suffer from high noise
levels, due to the shallow coverage in each sequencing unit and the extra experimental steps
required to preserve the locations of sequencing.

The goal of this project is to develop a deep generative model for spatial transcriptomics
data that will utilize the information from the physical locations of sequencing, and the tissue
organization reflected in corresponding pathology images to learn a latent representation of
the data which can be used for denoising the data. Several graph neural network models are
available for data imputation. Some of these models can be extended to incorporate the spatial graph for imputation. Standard datasets used by other imputation methods can be used for evaluation.

# CS690
Computational Genomics CS609 - Fall,2023 - IIT Kanpur, H.Zafar 
