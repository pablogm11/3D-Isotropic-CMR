from Utils import *
import vtk

patientPath= '/media/sf_VB_Folder/'
pathSave = 'case_01/results'
SA_name = 'case_01/s3D_IR-TFE_2BH_SENSE-1401_s3D_IR-TFE_2_BH_SENSE_20160711081415_1401_t682.nii'
LA_4CH_name = 'case_01/s3D_IR-TFE_BH_20slSENSE-1201_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1201_t681.nii'
LA_2CH_name  = 'case_01/s3D_IR-TFE_BH_20slSENSE-1301_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1301_t681.nii'
# regIm = mainRegScript(patientPath,SA_name,LA_4CH_name,LA_2CH_name,pathSave,typeRe = 'Rigid')
SA = ants.image_read(patientPath+SA_name)
SA = ants.resample_image(SA,[min(SA.spacing),min(SA.spacing),min(SA.spacing)])
LA_4CH = ants.image_read(patientPath+LA_4CH_name)
LA_2CH = ants.image_read(patientPath+LA_2CH_name)
readerSA = vtk.vtkNIFTIImageReader()
readerSA.SetFileName(patientPath+SA_name)
readerSA.Update()
reader4CH = vtk.vtkNIFTIImageReader()
reader4CH.SetFileName(patientPath+LA_4CH_name)
reader4CH.Update()
reader2CH = vtk.vtkNIFTIImageReader()
reader2CH.SetFileName(patientPath+LA_2CH_name)
reader2CH.Update()
print('j')