import numpy as np
import nibabel as nib
from nilearn import surface
from nilearn import datasets
from scipy.interpolate import griddata

class ToFlattenedSphere():
    def __init__(self,shape=(180,180)):
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
        self.shape = shape
        self._set_mapping()
        
    def _set_mapping(self):
        self.map_right = self._spherical2map(self._cartesian2spherical(surface.load_surf_mesh(self.fsaverage.infl_right)[0]),shape=self.shape)
        self.map_left = self._spherical2map(self._cartesian2spherical(surface.load_surf_mesh(self.fsaverage.infl_left)[0]),shape=self.shape)
        
    def _cartesian2spherical(self, xyz):
        #takes list xyz (single coord)
        x       = xyz[:,0]
        y       = xyz[:,1]
        z       = xyz[:,2]
        r       =  np.sqrt(x*x + y*y + z*z)
        elevation   =  np.arccos(z/r)  #to degrees
        azimuth     =  np.arctan2(y,x) 
        return np.array([r,elevation,azimuth]).transpose(1,0)
    
    def _spherical2map(self,spherical_coords,shape=(180,180)):
        rea = spherical_coords.copy()
        rea[:,1] /= np.pi
        rea[:,2] /= (2*np.pi)
        rea[:,2] += .5
        rea[:,1] *= shape[0]
        rea[:,2] *= shape[1]
        mapping = []
        for _,e,a in rea:
            e,a = int(e),int(a)
            mapping.append([e,a])

        mapping = np.array(mapping)
        return mapping
    
    def _make2dmap(self,texture,mapping,shape):
        
        grid_x, grid_y = np.mgrid[0:shape[0], 0:shape[1]]
        flatmap = griddata(mapping, texture, (grid_x, grid_y), method='nearest')
        
        return flatmap
    
    
    def _realign_flatmap(self,flatmap,mapping):
        
        return np.array([flatmap[e,a] for e,a in mapping])
        
    def _make_map(self,texture):
        grid_x, grid_y = np.mgrid[0:180, 0:180]
        
    def flatten(self,img):
        
        texture_right = surface.vol_to_surf(img, self.fsaverage.infl_right,inner_mesh=self.fsaverage.white_right)
        texture_right[np.isnan(texture_right)]=0
        
        flatmap_right = self._make2dmap(texture_right,self.map_right,shape=self.shape)
        texture_left = surface.vol_to_surf(img, self.fsaverage.infl_left,inner_mesh=self.fsaverage.white_left)
        texture_left[np.isnan(texture_left)]=0
        flatmap_left = self._make2dmap(texture_left,self.map_left,shape=self.shape)
        flatmaps = np.array([flatmap_right,flatmap_left]).transpose(1,2,0)
        #kernel = np.ones((3,3),np.float32)/9
        #flatmaps = cv2.filter2D(flatmaps, -1, kernel)
        return flatmaps
    
    def reconstruct(self,flatmaps):
        texture_right = self._realign_flatmap(flatmaps[:,0])
        texture_left = self._realign_flatmap(flatmaps[:,1])
        
        
        # affine??
        # revisit definition of affine
        # maybe the mesh file has it

# the cnn should have a large receptive field... (with dilated one?)