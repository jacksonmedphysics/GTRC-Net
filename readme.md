Code to infer tumour burden from post-therapy quantitative SPECT/CT imaging in Lu-177 PSMA treatment.
Input CT and QSPECT image and will designate malignant regions above specified funtional image threshold value.
Trainined based on SUV threshold of 3.0  but note that any similar threshold value can be selected by user and may perform comparably.
If qspect image is rescaled to units of SUV then enter that threshold value in the command line (eg 3.0) but can apply to non-scaled (Bq/ml units) image if appropriate threshold is known for the case (accounting for body weight/injected activity). Based on tensorflow with relatively lightweight processing footprint. Read requirements.txt for necessary python packages.

Usage:
python run_gtrc_inference.py -ct [path to ct image (*.nii or other ITK format)] -pet [path to qspect image] -suv [threshold value to apply to image] -out [output label path ('*.nii or other ITK format)

Includes the pre-trained model for fold #1 of the abstract submitted for SNMMI 2023: "Automated AI Tumor Burden Analysis on Lu-177 PSMA Quantitative SPECT with Global Threshold Regional Consensus Network (GTRC-Net)"
