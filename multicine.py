import numpy as np
import matplotlib.pyplot as plt
import ants
import os
from  pydicom import dcmread
from  pydicom.data import get_testdata_file 
from nipype import Node, Workflow
from nipype.interfaces.ants import N4BiasFieldCorrection
import ants
import nibabel as nib
import SimpleITK as sitk
from Utils import *

path = 'ED/rigid'
images_short = ants.image_read("/media/sf_VB_Folder/s3D_IR-TFE_2BH_SENSE-1401_s3D_IR-TFE_2_BH_SENSE_20160711081415_1401_t682.nii")
images_hLong = ants.image_read("s3D_IR-TFE_BH_20slSENSE-1201_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1201_t681.nii")
images_vLong = ants.image_read("/media/sf_VB_Folder/s3D_IR-TFE_BH_20slSENSE-1301_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1301_t681.nii")
imagCine = ants.image_read('/media/sf_VB_Folder/sBTFE_BH MFMULTICINE - as a 24 frames MultiVolume by ImagePositionPatientInstanceNumber.nrrd')
#Redifine to get only ED slices
arrC = imagCine.view()
arrC = arrC[23,:,:,:]
ED_img = ants.from_numpy(arrC,origin=imagCine.origin,spacing = imagCine.spacing,direction = imagCine.direction)


# ED_img = ants.resample_image(ED_img,[1.1429,1.1429,1.1429])
# ED_Re = ants.to_nibabel(ED_img)
# ED_Re.set_data_dtype('uint16')
# nib.save(ED_Re,'ED/ED_resampled.nii')

ED_imgCH = ED_img.view()
#Registration
regED_vl = ants.registration(ED_img,images_vLong,type_of_transform = "Rigid",aff_metric = 'mattes')
regED_hl = ants.registration(ED_img,images_hLong,type_of_transform = "Rigid",aff_metric = 'mattes')

cloneVL = ants.image_clone(images_vLong)
cloneHL = ants.image_clone(images_hLong)
roi_vl = overROI(cloneVL,regED_vl['fwdtransforms'],ED_img)
roi_hl = overROI(cloneHL,regED_hl['fwdtransforms'],ED_img)
roi_hlArr = roi_hl.view()
roi_hlArr = roi_hlArr == 1
roi_vlArr = roi_vl.view()
roi_vlArr = roi_vlArr == 1

print('ROI extraction Done')

newvl = ants.to_nibabel(regED_vl['warpedmovout'])
newvl.set_data_dtype('int16')
newhl = ants.to_nibabel(regED_hl['warpedmovout'])
newhl.set_data_dtype('int16')
nib.save(newvl,path+'/ED_1301.nii')
nib.save(newhl,path+'/ED_1201.nii')

#Normalization
# newvlCH = newvl.get_fdata()
newvlCH = regED_vl['warpedmovout'].view()
# newhlCH = newhl.get_fdata()
newhlCH = regED_hl['warpedmovout'].view()
for t in range(0,newhlCH.shape[2]):
    ED_imgCH[:,:,t] =  normOver(ED_imgCH[:,:,t],roi_hlArr[:,:,t]+roi_vlArr[:,:,t])
    newhlCH[:,:,t] = normOver(newhlCH[:,:,t],roi_hlArr[:,:,t])
    newvlCH[:,:,t] = normOver(newvlCH[:,:,t],roi_vlArr[:,:,t])



newvl = ants.to_nibabel(regED_vl['warpedmovout'])
newvl.set_data_dtype('int16')
newhl = ants.to_nibabel(regED_hl['warpedmovout'])
newhl.set_data_dtype('int16')
shortNorm = ants.to_nibabel(ED_img)
shortNorm.set_data_dtype('int16')
nib.save(newvl,path + '/ED_norm_1301.nii')
nib.save(newhl,path + '/ED_norm_1201.nii')
nib.save(shortNorm,path +'/ED_norm_1401.nii')
print('Normalization Done')
#Checkboard
hl_ch = np.zeros(ED_imgCH.shape)
vl_ch =np.zeros(ED_imgCH.shape)
for t in np.arange(0,ED_imgCH.shape[2]):
    hl_ch[:,:,t] = cheBoard(ED_imgCH[:,:,t],newhlCH[:,:,t],16,roi_hlArr[:,:,t])
    vl_ch[:,:,t] = cheBoard(ED_imgCH[:,:,t],newvlCH[:,:,t],16,roi_vlArr[:,:,t])
chest_vl = nib.Nifti1Image(vl_ch,newvl.affine,newvl.header)
chest_hl = nib.Nifti1Image(hl_ch,newhl.affine,newhl.header)
print("Checkboard filter applied")
nib.save(chest_hl,path + '/ED_chest_1201.nii')
nib.save(chest_vl,path + '/ED_chest_1301.nii')
