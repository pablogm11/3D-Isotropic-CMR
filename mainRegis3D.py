from Utils import *
import pyvista as pv

patientPath= '/media/sf_VB_Folder/'
#Patient 1799281
# pathSave = 'Anonymized - 1799281/results/'
# SA_name = 'Anonymized - 1799281/files/s3D_IR-TFE_BH_20sl_SENSE_-_1701_s3D_IR-TFE_BH_20sl_SENSE_20190918145547_1701_t803.nii'
# LA_4CH_name = 'Anonymized - 1799281/files/s3D_IR-TFE_BH_20sl_SENSE_-_1501_s3D_IR-TFE_BH_20sl_SENSE_20190918145547_1501_t912.nii'
# LA_2CH_name  = 'Anonymized - 1799281/files/s3D_IR-TFE_BH_20sl_SENSE_-_1801_s3D_IR-TFE_BH_20sl_SENSE_20190918145547_1801_t802.nii'

#Patient JMN0062H
pathSave = 'case_01/results'
SA_name = 'case_01/s3D_IR-TFE_2BH_SENSE-1401_s3D_IR-TFE_2_BH_SENSE_20160711081415_1401_t682.nii'
LA_4CH_name = 'case_01/s3D_IR-TFE_BH_20slSENSE-1201_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1201_t681.nii'
LA_2CH_name  = 'case_01/s3D_IR-TFE_BH_20slSENSE-1301_s3D_IR-TFE_BH_20sl_SENSE_20160711081415_1301_t681.nii'

regIm = mainRegScript(patientPath,SA_name,LA_4CH_name,LA_2CH_name,pathSave,typeRe = 'Rigid')
# print(regIm)
SA = regIm[0]
S_4CH = regIm[1]['warpedmovout']
S_2CH = regIm[2]['warpedmovout']
saView = SA.view()
ch4View = S_4CH.view()
ch2View = S_2CH.view()
SaVer = pv.wrap(saView)
ch4Ver = pv.wrap(ch4View)
ch2Ver = pv.wrap(ch2View)
p = pv.Plotter()
p.subplot(0,0)
p.add_volume(saView)
p.subplot(0,1)
p.add_volume(ch4View)
p.subplot(0,2)
p.add_volume(ch2View)
p.show()

