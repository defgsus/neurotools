from mvpa2.datasets.mri import fmri_dataset
import numpy as np

def vox_to_index(vox, dim):
    return int(vox[2] + dim[2] * (vox[1] + dim[1] * vox[0]))

def print_slice(fmri, z):
    dim = fmri.get_attr("voxel_dim")[0].value
    for y in range(dim[1]):
        print("".join("*" if fmri.samples[0][vox_to_index((x,y,z), dim)] else "." for x in range(dim[0])))

def get_combination(src, atlas):
    dim = src.get_attr("voxel_dim")[0].value
    transform = src.get_attr("imgaffine")[0].value
    inv_atlas = np.linalg.inv(atlas.get_attr("imgaffine")[0].value)
    atlas_dim = atlas.get_attr("voxel_dim")[0].value
    print("voxels x %s, atlas x %s" % (dim, atlas_dim))
    dic = {}
    list = []
    for z in range(dim[2]):
        #print "%1i/%2i" % (z + 1, dim[2])
        for y in range(dim[1]):
            for x in range(dim[0]):
                src_vox = [x,y,z,1]
                src_mni = transform.dot(src_vox)
                atlas_vox = inv_atlas.dot(src_mni)
                r = atlas.samples[0][vox_to_index(atlas_vox, atlas_dim)]
                vox = src.samples[0][ vox_to_index([x,y,z], dim) ]
                if r not in dic:
                    dic[r] = (1, vox)
                else:
                    dic[r] = (dic[r][0] + 1, dic[r][1] + vox)
                if r != 0:
                    list.append((r, src.samples[0][ vox_to_index([x,y,z], dim) ]))
                    #print(src_vox, src_mni, atlas_vox, r)
    x = [(r, dic[r][0], dic[r][1], float(dic[r][1]) / dic[r][0]) for r in dic]
    x = sorted(x, key=lambda x: x[3])
    return list, x
    #print("\n".join(str(r) for r in x))


def combine(atlas, fn):
    vox_fn = ".".join(fn.split(".")[:-1]) + "_region_map_voxels.txt"
    average_fn = ".".join(fn.split(".")[:-1]) + "_region_map_average.txt"
    print("sampling voxels per region for: %s\n"
          "output: %s\n"
          "output: %s\n" % (fn, vox_fn, average_fn))
    d = fmri_dataset(fn)
    vox, average = get_combination(d, atlas)
    print("writing output")
    with open(vox_fn, "w") as f:
        f.write("\n".join("\t".join(str(s) for s in v) for v in vox))
    with open(average_fn, "w") as f:
        f.write("\n".join("\t".join(str(s) for s in v) for v in average))

            #print_slice(d, d.get_attr("voxel_dim")[0].value[2]//2)


def cmdline(args):
    print("loading atlas")
    atlas = fmri_dataset("./IIT_GM_Desikan_atlas.nii.gz")
    # print_slice(atlas, atlas.get_attr("voxel_dim")[0].value[2]//2)

    for fn in args:
        combine(atlas, fn)


if __name__ == "__main__":
    import sys
    cmdline(sys.argv[1:])