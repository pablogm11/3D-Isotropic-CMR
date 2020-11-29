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

images_short = ants.image_read("/media/sf_VB_Folder/s3D_IR-TFE_2BH_SENSE-1401_s3D_IR-TFE_2_BH_SENSE_20160711081415_1401_t682.nii")
images_short = ants.resample_image(images_short,[1.1875,1.1875,1.1875])
images_hLong = ants.image_read("s3D_IR-TFE_BH_20slSENSE-1201_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1201_t681.nii")
# images_hLong = ants.resample_image(images_hLong,[320,320,320],True)
images_vLong = ants.image_read("/media/sf_VB_Folder/s3D_IR-TFE_BH_20slSENSE-1301_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1301_t681.nii")
# images_vLong = ants.resample_image(images_vLong,[320,320,320],True)

shortO = ants.to_nibabel(images_short)
vLongO = ants.to_nibabel(images_vLong)
hLongO = ants.to_nibabel(images_hLong)
shortO.set_data_dtype('int16')
vLongO.set_data_dtype('int16')
hLongO.set_data_dtype('int16')
nib.save(shortO,'reSampled_1401.nii')
nib.save(vLongO,'reSampled_1301.nii')
nib.save(hLongO,'reSampled_1201.nii')
#Save initial chests
# shortCH = shortO.get_fdata()
shortCH = images_short.view()


#Registration
reg1 = ants.registration(images_short,images_vLong,type_of_transform = "Affine",aff_metric = 'mattes')
reg2 = ants.registration(images_short,images_hLong,type_of_transform = "Affine",aff_metric = 'mattes')
print('Registration Done')

#ROI extraction
cloneVL = ants.image_clone(images_vLong)
cloneHL = ants.image_clone(images_hLong)
roi_vl = overROI(cloneVL,reg1['fwdtransforms'],images_short)
roi_hl = overROI(cloneHL,reg2['fwdtransforms'],images_short)
roi_hlArr = roi_hl.view()
roi_hlArr = roi_hlArr == 1
roi_vlArr = roi_vl.view()
roi_vlArr = roi_vlArr == 1

print('ROI extraction Done')

newvl = ants.to_nibabel(reg1['warpedmovout'])
newvl.set_data_dtype('int16')
newhl = ants.to_nibabel(reg2['warpedmovout'])
newhl.set_data_dtype('int16')
nib.save(newvl,'1301.nii')
nib.save(newhl,'1201.nii')

#Normalization
# newvlCH = newvl.get_fdata()
newvlCH = reg1['warpedmovout'].view()
# newhlCH = newhl.get_fdata()
newhlCH = reg2['warpedmovout'].view()
for t in range(0,newhlCH.shape[1]):
    shortCH[:,t,:] = normOver(shortCH[:,t,:],roi_hlArr[:,t,:]+roi_vlArr[:,t,:])
    newhlCH[:,t,:] = normOver(newhlCH[:,t,:],roi_hlArr[:,t,:])
    newvlCH[:,t,:] = normOver(newvlCH[:,t,:],roi_vlArr[:,t,:])

# newvl = nib.Nifti1Image(newvlCH,newvl.affine,newvl.header)
# newhl = nib.Nifti1Image(newhlCH,newhl.affine,newhl.header)
newvl = ants.to_nibabel(reg1['warpedmovout'])
newvl.set_data_dtype('int16')
newhl = ants.to_nibabel(reg2['warpedmovout'])
newhl.set_data_dtype('int16')
shortNorm = ants.to_nibabel(images_short)
shortNorm.set_data_dtype('int16')
nib.save(newvl,'norm_1301.nii')
nib.save(newhl,'norm_1201.nii')
nib.save(shortNorm,'norm_1401.nii')
print('Normalization Done')
#Checkboard
hl_ch = np.zeros(shortCH.shape)
vl_ch =np.zeros(shortCH.shape)
for t in np.arange(0,shortCH.shape[1]):
    hl_ch[:,t,:] = cheBoard(shortCH[:,t,:],newhlCH[:,t,:],8,roi_hlArr[:,t,:])
    vl_ch[:,t,:] = cheBoard(shortCH[:,t,:],newvlCH[:,t,:],8,roi_vlArr[:,t,:])
chest_vl = nib.Nifti1Image(vl_ch,newvl.affine,newvl.header)
chest_hl = nib.Nifti1Image(hl_ch,newhl.affine,newhl.header)
print("Checkboard filter applied")
nib.save(chest_hl,'chest_1201.nii')
nib.save(chest_vl,'chest_1301.nii')
# ants.image_write(newvl,'1401.nii')
# ants.image_write(newhl,'1301.nii')
print("Done")
#dir_path = "/media/sf_VB_Folder"
# directories = os.walk(dir_path)
# images_path = []
# folder_name = []
# for root, dirs, files in directories:
#     cases = [] 
#     state = 0 
#     for name in files:
#         if name.endswith(".dcm"):
#             ima = dcmread(os.path.join(root,name))
#             case_ima = np.array(ima.pixel_array)
#             cases.append(case_ima.astype("float64"))
#             state = 1
#     if state == 1:
#         images_path.append(cases[:])
#         folder_name.append(root.replace(dir_path,""))
# for i in np.arange(0,len(images_path)):
#     count = 1
#     nplots = len(images_path[i])
#     rows = 3
#     col = round((nplots+1)/rows)
#     plt.figure(i)
#     for j in images_path[i]:
#         image = j   
#         plt.subplot(rows,col,count)
#         plt.imshow(image, cmap = "gray")
#         plt.axis("off")
#         count =count + 1
#     plt.suptitle(folder_name[i].replace("/",""))   
#     plt.show()
# x = images_path[2][2]
# plt.figure()
# plt.imshow(images_path[2][2],cmap = "gray")
# plt.show()
# x_corre = ants.n4_bias_field_correction(ants.from_numpy(x))
# plt.figure(2)
# ants.plot(x_corre)

