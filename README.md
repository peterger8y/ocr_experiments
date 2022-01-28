Just a quick read-me to get this repo working for ya...

Run this snippet in terminal to test out the origami model on a single image-
Reach out for the .env the dataset to run the algo. 

-> python origami_ocr/testing_model_transcriptions.py -ip <image_filename_here>

excluding the -ip argument (location of image for desired transcription) instruct
the algorithm to look for a 'dataset' folder inside of the origami_ocr subfolder. 
If it doesnt find one, It'll throw an error.

Having the image file location handy gives you string output of that file. excluding it
gives you metrics about how well the model is transcribing data. 

=======================================================================================
DATA AUGMENTATION:

check out data generation augmentation for the algorithm that creates novel data using 
pre_sliced images for bounding box placement.