import numpy as np
import matplotlib.pyplot as plt
import ants
import os
from  pydicom import dcmread
from  pydicom.data import get_testdata_file 
from nipype import Node, Workflow
from nipype.interfaces.ants import N4BiasFieldCorrection
import nibabel as nib
import SimpleITK as sitk

def cheBoard (img1, img2, samples,roi):
    count = 0
    x = img1.shape[0]
    y = img1.shape[1]
    ch = np.zeros([x,y])
    for i in range(0,x,int(x/samples)):
        count += 1
        for j in range(0,y,int(y/samples)):
            count += 1
            if count % 2 == 0:
                
                ch[i:i+int(x/samples)-1,j:j+int(y/samples)-1]=1
            else:
                
                ch[i:i+int(x/samples)-1,j:j+int(y/samples)-1]=0
    
    f = np.where(roi == False)
    for t in range(0,len(f[0])):
        ch[f[0][t],f[1][t]] = 0
    invCh = np.ones([x,y]) - ch

    moving = ch*img2
    fixed = invCh*img1
    final = moving + fixed
    return final

def normOver (img1,roi):
    mean1 = np.mean(img1[roi])
    std1 = np.std(img1[roi])
    # mean2 = np.mean(img2[roi])
    # std2 = np.std(img2[roi])
    nImg1 = (img1 - mean1)/std1
    # nImg2 = (img2 - mean2)/std2
    return nImg1#, nImg2
def overROI (images,transformation,fixed):
    
    nr = images.view()
    nr[:,:,:] = np.ones(images.shape)
    # nr = np.ones(images.shape)
    roi = ants.apply_transforms(fixed,images,transformation)
    return roi          
def mainRegScript(patientPath,SA_name,LA_4CH_name,LA_2CH_name,pathSave,typeRe):
    #Load initial images
    SA = ants.image_read(patientPath+SA_name)
    LA_4CH = ants.image_read(patientPath+LA_4CH_name)
    LA_2CH = ants.image_read(patientPath+LA_2CH_name)

    #Registration
    regSA_4CH = ants.registration(SA,LA_4CH,type_of_transform = typeRe,aff_metric = 'mattes')
    regSA_2CH = ants.registration(SA,LA_2CH,type_of_transform = typeRe,aff_metric = 'mattes')
    print('Registration Done')

    #ROI extraction
    clone4CH = ants.image_clone(LA_4CH)
    clone2CH = ants.image_clone(LA_2CH)
    roi_4CH = overROI(clone4CH,regSA_4CH['fwdtransforms'],SA)
    roi_2CH = overROI(clone2CH,regSA_2CH['fwdtransforms'],SA)
    roi_4CHArr = roi_4CH.view()
    roi_4CHArr = roi_4CHArr == 1
    roi_2CHArr = roi_2CH.view()
    roi_2CHArr = roi_2CHArr == 1

    print('ROI extraction Done')

    new4CH = ants.to_nibabel(regSA_4CH['warpedmovout'])
    new4CH.set_data_dtype('int16')
    new2CH = ants.to_nibabel(regSA_2CH['warpedmovout'])
    new2CH.set_data_dtype('int16')
    nib.save(new4CH,pathSave + '4CH.nii')
    nib.save(new2CH,pathSave + '2CH.nii')

    #Normalization

    new4CH_CH = regSA_4CH['warpedmovout'].view()
    new2CH_CH = regSA_2CH['warpedmovout'].view()
    short_CH = SA.view()
    for t in range(0,new2CH_CH.shape[2]):
        short_CH[:,:,t] = normOver(short_CH[:,:,t],roi_4CHArr[:,:,t]+roi_2CHArr[:,:,t])
        new4CH_CH[:,:,t] = normOver(new4CH_CH[:,:,t],roi_4CHArr[:,:,t])
        new2CH_CH[:,:,t] = normOver(new2CH_CH[:,:,t],roi_2CHArr[:,:,t])


    new4CH = ants.to_nibabel(regSA_4CH['warpedmovout'])
    new4CH.set_data_dtype('int16')
    new2CH = ants.to_nibabel(regSA_2CH['warpedmovout'])
    new2CH.set_data_dtype('int16')
    shortNorm = ants.to_nibabel(SA)
    shortNorm.set_data_dtype('int16')
    nib.save(new4CH,pathSave + 'norm_4CH.nii')
    nib.save(new2CH,pathSave + 'norm_2CH.nii')
    nib.save(shortNorm,pathSave + 'norm_SA.nii')
    print('Normalization Done')

    #Checkboard
    ch_4CH = np.zeros(short_CH.shape)
    ch_2CH =np.zeros(short_CH.shape)
    for t in np.arange(0,short_CH.shape[2]):
        ch_4CH[:,:,t] = cheBoard(short_CH[:,:,t],new4CH_CH[:,:,t],16,roi_4CHArr[:,:,t])
        ch_2CH[:,:,t] = cheBoard(short_CH[:,:,t],new2CH_CH[:,:,t],16,roi_2CHArr[:,:,t])
    chest_4CH = nib.Nifti1Image(ch_4CH,new4CH.affine,new4CH.header)
    chest_2CH = nib.Nifti1Image(ch_2CH,new2CH.affine,new2CH.header)
    print("Checkboard filter applied")
    nib.save(chest_4CH,pathSave + 'chest_4CH.nii')
    nib.save(chest_2CH,pathSave + 'chest_2CH.nii')
    print("Done")

