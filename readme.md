Code to infer tumour burden from post-therapy quantitative SPECT/CT imaging from Lu-177 PSMA treatment.
Input CT and QSPECT image. Trainined based on SUV threshold of 3.0  but note that any similar threshold value can be selected by user and may perform comparably.
If qspect image is rescaled to units of SUV then enter that threshold value in the command line (eg 3.0) but can as easily apply to non-scaled Bq/ml unit image if appropriate threshold is known for the case (accounting for body weight/injected activity)


Usage:
python run_gtrc_inference.py -ct [path to ct image (*.nii or other ITK format)] -pet [path to qspect image] -suv [threshold value to apply to image] -out [output label path ('*.nii or other ITK format)

