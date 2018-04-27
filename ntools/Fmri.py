import nibabel as nib
import numpy as np


class Fmri:

    def __init__(self, filename=None):
        self.image = None
        self._data = None
        self.filename = None
        if filename:
            self.load(filename)

    def load(self, filename):
        self._data = None
        self.image = nib.load(filename)
        self.filename = filename

    @property
    def shape(self):
        return self.image.header.get_data_shape()

    @property
    def dtype(self):
        return self.image.header.get_data_dtype()

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def depth(self):
        return self.shape[2]

    @property
    def affine(self):
        return self.image.get_affine()

    @property
    def inverse_affine(self):
        return np.linalg.inv(self.affine)

    @property
    def header(self):
        return self.image.header

    @property
    def data(self):
        if self._data is None:
            self._data = self.image.dataobj.get_unscaled()
        return self._data

    def voxel(self, x, y=None, z=None):
        if y is None:
            return self.data[int(x[0]), int(x[1]), int(x[2])]
        return self.data[int(x), int(y), int(z)]

    def print_slice(self, z):
        chars = ".:+*#"
        numchars = len(chars)
        mi, ma = np.min(self.data), np.max(self.data)
        for y in range(self.height):
            print("".join(chars[min(numchars-1, int((self.voxel(x, y, z)-mi)/(ma-mi)*numchars))]
                          for x in range(self.width)))


if __name__ == "__main__":

    #fmri = Fmri("./IIT_GM_Desikan_atlas.nii.gz")
    fmri = Fmri("./adni_brain.nii")
    print(fmri.shape)
    print(fmri.affine)
    #print(fmri.image.dataobj.get_unscaled())
    fmri.print_slice(fmri.depth//2)