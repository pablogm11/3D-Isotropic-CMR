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

path = 'ED/2D_3D'
images_short = ants.image_read("/media/sf_VB_Folder/s3D_IR-TFE_2BH_SENSE-1401_s3D_IR-TFE_2_BH_SENSE_20160711081415_1401_t682.nii")
images_hLong = ants.image_read("s3D_IR-TFE_BH_20slSENSE-1201_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1201_t681.nii")
images_vLong = ants.image_read("/media/sf_VB_Folder/s3D_IR-TFE_BH_20slSENSE-1301_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1301_t681.nii")
imagCine = ants.image_read('/media/sf_VB_Folder/sBTFE_BH MFMULTICINE - as a 24 frames MultiVolume by ImagePositionPatientInstanceNumber.nrrd')
#Redifine to get only ED slices
arrC = imagCine.view()
arrC = arrC[23,:,:,:]
ED_img = ants.from_numpy(arrC,origin=imagCine.origin,spacing = imagCine.spacing,direction = imagCine.direction)
final = []
outRoi = []
for x in range(0,images_vLong.view().shape[0]):
    #Clone SA
    ED_imgCL = ants.image_clone(ED_img)
    ED_imgCH = ED_imgCL.view()
    #Create Files
    addPa = '/'+str(x)
    #Get 2D slice
    arrVL = images_vLong.view()
    arrVL = arrVL[x,:,:]
    VL_new = ants.from_numpy(np.reshape(arrVL,[1,288,288]),origin=images_vLong.origin,spacing = images_vLong.spacing,direction = images_vLong.direction)
    VL_new2 = ants.to_nibabel(VL_new)
    nib.save(VL_new2,path+addPa+'BR_ED_1301.nii')
    #Registration
    regED_vl = ants.registration(ED_imgCL,VL_new,type_of_transform = "Affine",aff_metric = 'mattes')
    # regED_hl = ants.registration(ED_img,images_hLong,type_of_transform = "Affine",aff_metric = 'mattes')

    cloneVL = ants.image_clone(VL_new)
    # cloneHL = ants.image_clone(images_hLong)
    roi_vl = overROI(cloneVL,regED_vl['fwdtransforms'],ED_imgCL)
    # roi_hl = overROI(cloneHL,regED_hl['fwdtransforms'],ED_img)
    # roi_hlArr = roi_hl.view()
    # roi_hlArr = roi_hlArr == 1
    outRoi.append(roi_vl)
    roi_vlArr = roi_vl.view()
    roi_vlArr = roi_vlArr == 1

    print('ROI extraction Done')

    newvl = ants.to_nibabel(regED_vl['warpedmovout'])
    newvl.set_data_dtype('int16')
    # newhl = ants.to_nibabel(regED_hl['warpedmovout'])
    # newhl.set_data_dtype('int16')
    nib.save(newvl,path+addPa+'ED_1301.nii')
    # nib.save(newhl,path+'/ED_1201.nii')

    #Normalization
    # newvlCH = newvl.get_fdata()
    newvlCH = regED_vl['warpedmovout'].view()
    # newhlCH = newhl.get_fdata()
    # newhlCH = regED_hl['warpedmovout'].view()
    for t in range(0,newvlCH.shape[2]):
        ED_imgCH[:,:,t] =  normOver(ED_imgCH[:,:,t],roi_vlArr[:,:,t])
        # newhlCH[:,:,t] = normOver(newhlCH[:,:,t],roi_hlArr[:,:,t])
        newvlCH[:,:,t] = normOver(newvlCH[:,:,t],roi_vlArr[:,:,t])



    newvl = ants.to_nibabel(regED_vl['warpedmovout'])
    newvl.set_data_dtype('int16')
    # newhl = ants.to_nibabel(regED_hl['warpedmovout'])
    # newhl.set_data_dtype('int16')
    shortNorm = ants.to_nibabel(ED_imgCL)
    shortNorm.set_data_dtype('int16')
    nib.save(newvl,path +addPa+'ED_norm_1301.nii')
    # nib.save(newhl,path + '/ED_norm_1201.nii')
    nib.save(shortNorm,path +addPa+'ED_norm_1401.nii')
    print('Normalization Done')
    #Checkboard
    # hl_ch = np.zeros(ED_imgCH.shape)
    vl_ch =np.zeros(ED_imgCH.shape)
    for t in np.arange(0,ED_imgCH.shape[2]):
        # hl_ch[:,:,t] = cheBoard(ED_imgCH[:,:,t],newhlCH[:,:,t],16,roi_hlArr[:,:,t])
        vl_ch[:,:,t] = cheBoard(ED_imgCH[:,:,t],newvlCH[:,:,t],16,roi_vlArr[:,:,t])
    chest_vl = nib.Nifti1Image(vl_ch,newvl.affine,newvl.header)
    # chest_hl = nib.Nifti1Image(hl_ch,newhl.affine,newhl.header)
    print("Checkboard filter applied")
    # nib.save(chest_hl,path + '/ED_chest_1201.nii')
    nib.save(chest_vl,path +addPa+'ED_chest_1301.nii')
    final.append(regED_vl['warpedmovout'])
finaOut = ants.merge_channels(final)
finalroi = ants.merge_channels(outRoi)
finalroiArr = finalroi.view()
finaOutArr = finaOut.view()
out = np.zeros(ED_imgCL.shape)
for i in range(0,finaOut.components):
    slice = np.zeros([ED_imgCH.shape[0],ED_imgCH.shape[1]])

    for j in range(0,finaOut.shape[2]):
        slice =  slice + finaOutArr[i,:,:,j]*finalroiArr[i,:,:,j]

    out[:,:,i] = slice
out = ants.from_numpy(out,origin=finaOut.origin,spacing = finaOut.spacing,direction = finaOut.direction)
out = ants.to_nibabel(out)
nib.save(out,path+'/ED_final.nii')
print("Quieto ahi")