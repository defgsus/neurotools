#!/bin/python

from nifti import NiftiImage
import numpy as np
import os

def to_mni_coord(src, voxel):
    v = np.array(voxel) 
    return src.sform.dot( v.tolist() + [1] )[:3] 

def to_voxel_coord(src, coord):
    v = np.array(coord) 
    return np.linalg.inv(src.sform).dot( v.tolist() + [1] )[:3] 

def to_mni_coord_and_val(src, voxel):
    return to_mni_coord(src, voxel).tolist() + [ src.data[voxel[2]][voxel[1]][voxel[0]] ]

def print_combine(src, regions, out_file):
    inv_regions = np.linalg.inv(regions.sform)
    for z in range(src.extent[2]): 
        print "%1i/%2i" % (z+1, src.extent[2])
        for y in range(src.extent[1]): 
            for x in range(src.extent[0]): 
                srcvox = [x,y,z,1]
                srcmni = src.sform.dot(srcvox)
                rvox = inv_regions.dot(srcmni)
                r = regions.data[int(rvox[2])][int(rvox[1])][int(rvox[0])]
                if r != 0:  
                    #print >> out_file, "%1i\t%2r" % (r, src.data[z][y][x])
                    print "%1i\t%2r" % (r, src.data[z][y][x])
                    print(srcvox, srcmni, rvox, r)
        if z == 11:
            return

regions = NiftiImage("IIT_GM_Desikan_atlas")
src = NiftiImage("SYS_Parents_female_PCA(all)_features2mm_GM_p001")
#src = NiftiImage("IIT_GM_Desikan_atlas.nii.gz")
print_combine(src, regions, None)
#print(src.data[50][70][35], src.data[70][34][55])
# (-0.45185664, -0.44627368)
def combine_directory():
    for path, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.img'):
                fn = os.path.join(path, f)[:-4]
                tablefn = os.path.join(path, 'combined/' + f[:-4] + '_per_region_desikan.txt')
                print fn
                outfile = open(tablefn, 'wt')
                src = NiftiImage(fn)
                print_combine(src, regions, outfile)

#combine_directory()
