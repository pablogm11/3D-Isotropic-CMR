from Utils import *
patientPath= '/media/sf_VB_Folder/'
#Patient 1799281
# pathSave = 'Anonymized - 1799281/results/'
# SA_name = 'Anonymized - 1799281/files/s3D_IR-TFE_BH_20sl_SENSE_-_1701_s3D_IR-TFE_BH_20sl_SENSE_20190918145547_1701_t803.nii'
# LA_4CH_name = 'Anonymized - 1799281/files/s3D_IR-TFE_BH_20sl_SENSE_-_1501_s3D_IR-TFE_BH_20sl_SENSE_20190918145547_1501_t912.nii'
# LA_2CH_name  = 'Anonymized - 1799281/files/s3D_IR-TFE_BH_20sl_SENSE_-_1801_s3D_IR-TFE_BH_20sl_SENSE_20190918145547_1801_t802.nii'

#Patient JMN0062H
pathSave = 'Anonymized - JMN0062H/results/'
SA_name = 'Anonymized - JMN0062H/files/tfi65_psir_t1_EJE_CORTO_MAG_-_46_tfi65_psir_t1_EJE_CORTO_20201111161048_46.nii'
LA_4CH_name = 'Anonymized - JMN0062H/files/LGE_2D_4CH_MAG_-_56_LGE_2D_4CH_20201111161048_56.nii'
LA_2CH_name  = 'Anonymized - JMN0062H/files/LGE_2D_2CH_MAG_-_54_LGE_2D_2CH_20201111161048_54.nii'

mainRegScript(patientPath,SA_name,LA_4CH_name,LA_2CH_name,pathSave,typeRe = 'Affine')
