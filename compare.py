from mvpa2.datasets.mri import fmri_dataset
import numpy as np


def mvpa():
    d = fmri_dataset("SYS_Parents_female_PCA(all)_features2mm_GM_p001.img")
    dim = d.get_attr("voxel_dim")[0].value
    transform = d.get_attr("imgaffine")[0].value
    print(dim, transform)

    print(transform.dot([1,2,3,1]))




mvpa()
