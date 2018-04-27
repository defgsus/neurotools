import sys, os
import argparse
#from threading import current_thread
from multiprocessing.pool import ThreadPool

import numpy as np

from ntools import *


class Global:
    _atlas = None

    @classmethod
    def atlas(cls):
        if cls._atlas is None:
            cls.load()
        return cls._atlas

    @classmethod
    def load(cls, fn=None):
        fn = fn or "./IIT_GM_Desikan_atlas.nii.gz"
        cls._atlas = Fmri(fn)


def get_combination(src, atlas):
    dim = src.shape
    transform = src.affine
    inv_atlas = atlas.inverse_affine
    atlas_dim = atlas.shape
    print("brain x %s (%s, max=%s, av=%s), atlas x %s (%s, max=%s)" % (
        dim, src.dtype, np.max(src.data), round(np.average(src.data), 2),
        atlas_dim, atlas.dtype, np.max(atlas.data),
    ))

    max_val = None
    if src.dtype == "uint8":
        max_val = 255

    dic = {}
    list = []
    for z in range(dim[2]):
        #print "%1i/%2i" % (z + 1, dim[2])
        for y in range(dim[1]):
            for x in range(dim[0]):
                # map src voxel to atlas voxel
                src_vox = [x, y, z, 1]
                src_mni = transform.dot(src_vox)
                atlas_vox = inv_atlas.dot(src_mni)

                region = atlas.voxel(atlas_vox)
                weight = float(src.voxel(x, y, z)) / max_val

                # count weight for region
                if region not in dic:
                    dic[region] = (1, weight)
                else:
                    dic[region] = (dic[region][0] + 1, dic[region][1] + weight)

                if region != 0:
                    list.append((region, weight))
                    #print(src_vox, src_mni, atlas_vox, region, weight)

    # average per region
    av = [(r, dic[r][0], dic[r][1], float(dic[r][1]) / dic[r][0]) for r in dic]
    av.sort(key=lambda x: x[3])

    return list, av


def combine(fn, output_dir=None):
    out_fn = fn
    if output_dir is not None:
        out_fn = os.path.join(output_dir, fn.split(os.path.sep)[-1])

    vox_fn = ".".join(out_fn.split(".")[:-1]) + "_region_map_voxels.txt"
    average_fn = ".".join(out_fn.split(".")[:-1]) + "_region_map_average.txt"
    print("sampling voxels per region for: %s\n"
          "output: %s\n"
          "output: %s\n" % (fn, vox_fn, average_fn))
    d = Fmri(fn)

    vox, average = get_combination(d, Global.atlas())

    print("writing output")
    with open(vox_fn, "w") as f:
        f.write("\n".join("\t".join(str(round(s, 5)) for s in v) for v in vox))
    with open(average_fn, "w") as f:
        f.write("\n".join("\t".join(str(round(s, 5)) for s in v) for v in average))


def process_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", nargs="?", const=None, type=str,
                        help="""Define output directory for processed files. 
                                If omitted, output files will be placed besides the input files""")
    parser.add_argument("-j", nargs="?", type=int, default=0,
                        help="""Number of threads. No multithreading by default""")
    parser.add_argument("files", nargs="+", type=str,
                        help="List of input nifti files to process")
    options = parser.parse_args(args)

    if not options.files:
        parser.print_help()
        exit(1)

    if options.j == 0:
        for i, fn in enumerate(options.files):
            print("--- %s of %s ---" % (i, len(options.files)))
            combine(fn, options.d)
    else:
        def _func(fn):
            combine(fn, options.d)

        pool = ThreadPool(options.j)
        pool.map(_func, options.files)
        pool.close()
        pool.join()


if __name__ == "__main__":
    #print(sys.argv)
    process_arguments(sys.argv[1:])

    #cmdline(sys.argv[1:])
    #combine("./adni_brain.nii")
    #combine("./rp1fam1002_427025_20070113_100405_201_mri_affine.nii")
