import pandas as pd
import tensorflow.keras as keras
import numpy as np
import os
from os.path import join
import SimpleITK as sitk
from scipy import ndimage
import tensorflow as tf

from tensorflow.keras.models import load_model
from gtrc.file_loader import load_cropped_inference
from gtrc.gtrc_utils import create_subregion_labels

import argparse
parser=argparse.ArgumentParser(description='derives training information for consensus optimisation')
parser.add_argument('--ct_path', '-ct', type=str, help='Location of CT image (nifti/ITK format)',required=True)
parser.add_argument('--pet_path', '-pet', type=str, help='Location of PET/QSPECT image (nifti/ITK format)',required=True)
parser.add_argument('--output_path','-out',type=str,help='Location of output file, default "predicted_ttb.nii.gz',default='predicted_ttb.nii.gz',required=False)
parser.add_argument('--suv_threshold','-suv',type=float,default=3.0,help='SUV or Bq/ml threshold to apply to functional image (default=3.0)',required=False)
parser.add_argument('--fold_number','-fold',type=int,default=1,help='which fold to use #s 1-5',required=False)
args=parser.parse_args()

ct_path=args.ct_path
pet_path=args.pet_path
suv_threshold=args.suv_threshold
fold_number=args.fold_number #read in fold number from command line (which subdirectory to evaluate in training dir)
output_path=args.output_path


train_dir='training' #folder to output training/preprocessing information and models
fold_dir=join(train_dir,'fold_'+str(fold_number).zfill(2))
df_res=pd.read_csv(join(train_dir,'training_resolution.csv'),index_col=0)
resample_extent=df_res.values[0,:3]
resample_dimensions=np.flip(df_res.values[0,3:6].astype(int))
subregion_local_max=df_res.subregion_local_max.values[0]
subregion_radius=df_res.subregion_radius.values[0]

consensus_model_path=join(fold_dir,'consensus_model.hdf5')
consensus_model=load_model(consensus_model_path,compile=False)
for f in os.listdir(fold_dir): #find model files for tumor and normal regions
    if f.startswith('GTRCNet') and f.endswith('.hdf5') and 'normal' in f: #os.path.isdir(join(fold_dir,f))
        norm_model_path=join(fold_dir,f)
        norm_model=load_model(norm_model_path,compile=False)
        if 'QSCT' in f:
            include_ct=True
            input_depth=2
        else:
            include_ct=False
            input_depth=1
    elif f.startswith('GTRCNet') and f.endswith('.hdf5') and 'tumor' in f:
        ttb_model_path=join(fold_dir,f)
        ttb_model=load_model(ttb_model_path,compile=False)

include_background_channel=True
if include_background_channel:
    label_channel=1
else:
    label_channel=0

qs=sitk.ReadImage(pet_path)
ct=sitk.ReadImage(ct_path)
x,resampled=load_cropped_inference(resample_extent, resample_dimensions, pet_path, ct_path, batch_shape=True) #load data into tf inference format
x=x.astype('float32')
x[...,0]=x[...,0]/suv_threshold  #rescale PET/QSPECT channel to threshold=1.0

rs=sitk.ResampleImageFilter() #create resampler to get intermediate files back to original resolution
rs.SetInterpolator(sitk.sitkLinear)
rs.SetReferenceImage(qs)

qs_ar=sitk.GetArrayFromImage(qs) #create numpy arrays for scoring
suv_ar=qs_ar/suv_threshold
qs_suv=sitk.GetImageFromArray(suv_ar)
qs_suv.CopyInformation(qs)
ct=rs.Execute(ct)
ct_ar=sitk.GetArrayFromImage(ct)
threshold_ar=(suv_ar>=1.0) #get all image volume above global threshold

print('Running Tumour Region inference')
pred_ttb_resampled_ar=ttb_model(np.expand_dims(x,0)).numpy()[0,...,label_channel] #infer ttb label. Convert to numpy and take only label channel
pred_ttb_resampled_im=sitk.GetImageFromArray(pred_ttb_resampled_ar) #create SITK image and associate predicted array with U-Net spatial information
pred_ttb_resampled_im.CopyInformation(resampled)   
pred_ttb_original_im=rs.Execute(pred_ttb_resampled_im) #resample to original input resolution (linear, continuous [0.0-1.0] prediction)
pred_ttb_ar=(sitk.GetArrayFromImage(pred_ttb_original_im)>0.5).astype('int8') #clip values >0.5 to binary array
pred_ttb_original_im=sitk.GetImageFromArray(pred_ttb_ar) #create SITK image and copy spatial information from TTB label
pred_ttb_original_im.CopyInformation(qs)

print('Running Physiological Region inference')
pred_norm_resampled_ar=norm_model(np.expand_dims(x,0)).numpy()[0,...,label_channel] #same as above but for normal UNet prediction
pred_norm_resampled_im=sitk.GetImageFromArray(pred_norm_resampled_ar)
pred_norm_resampled_im.CopyInformation(resampled)
pred_norm_original_im=rs.Execute(pred_norm_resampled_im)
pred_norm_ar=(sitk.GetArrayFromImage(pred_norm_original_im)>0.5).astype('int8')
pred_norm_original_im=sitk.GetImageFromArray(pred_norm_ar)
pred_norm_original_im.CopyInformation(qs)


qs_spacing=qs.GetSpacing() #get voxel spacing and volume for analysis
voxel_volume=np.prod(np.array(qs_spacing))/1000. #in ml
ws_im=create_subregion_labels(qs_suv,threshold_ar,subregion_local_max,subregion_radius)
print('Determining candidate subregions')
labels_ws=sitk.GetArrayFromImage(ws_im) #convert subregion image to numpy array
total_subregions=int(labels_ws.max()) ##get max value from subregion array for iterating (note some may be empty already)
print(total_subregions,'subregions detected...')
ttb_ar=np.zeros(qs_ar.shape)
n_included=0
print('Running consensus evaluation')
for i in range(total_subregions): #iterate through the subregions
    total_voxels=(labels_ws==(i+1)).sum() #count number of voxels in region
    total_volume=total_voxels*voxel_volume #convert to volume (ml/cc)
    if total_volume>0.: #if any volume found compute basic stats
        suv_max=qs_ar[labels_ws==(i+1)].max() #qspect SUV max
        suv_mean=qs_ar[labels_ws==(i+1)].mean() #qspect SUV mean
        if include_ct:
            ct_hu_mean=ct_ar[labels_ws==(i+1)].mean() #ct Hounsfield Unit mean
        else:
            ct_hu_mean=0. #set to 0 if CT omitted
        pred_ttb_overlap=(np.logical_and((labels_ws==(i+1)),(pred_ttb_ar>0.5)).sum())/total_voxels
        pred_norm_overlap=(np.logical_and((labels_ws==(i+1)),(pred_norm_ar>0.5)).sum())/total_voxels
        consensus_x=np.expand_dims(np.array([total_volume,suv_max,suv_mean,ct_hu_mean,pred_ttb_overlap,pred_norm_overlap]),0)
        consensus_pred=consensus_model(consensus_x).numpy()[0][0] #get prediction [0.0-1.0]
        if consensus_pred>=0.5: #if better than 0.5 include subregion
            n_included+=1
            ttb_ar[labels_ws==(i+1)]=n_included #initially assign each subregion its own value...
print(n_included,'/',total_subregions,'candidate regions included')
ttb_ar=(ttb_ar>0).astype('int16')
ttb_im=sitk.GetImageFromArray(ttb_ar) #save consensus array to image and copy spatial information
ttb_im.CopyInformation(qs)
ttb_im=sitk.Cast(ttb_im,sitk.sitkInt16)
sitk.WriteImage(ttb_im,output_path)
