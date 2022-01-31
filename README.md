Just a quick read-me to get this repo working for ya...

Run this snippet in terminal to test out the origami model on a single image-
Reach out for the .env the dataset to run the algo. 

-> python origami_ocr/testing_model_transcriptions.py -ip <image_filename_here>

excluding the -ip argument (location of image for transcription) instruct
the algorithm to look for a 'dataset' folder inside of the origami_ocr subfolder. 
If it doesnt find one, It'll throw an error.

Having the image file location handy gives you string output of that file. excluding it
gives you metrics about how well the model is transcribing data. 

=======================================================================================
DATA AUGMENTATION:

check out data generation augmentation for the algorithm that creates novel data using 
pre_sliced images for bounding box placement.... Right now, it just places words randomly
on a page... Nothing exciting yet, but more to come!

========================================================================================
BOUNDING BOXES

The Boudning box algorithm may be the trickiest to get working for you.... This has to do with
its use of something called lanms, or locality aware non maximal suppression. It is faster than 
vanilla non-maximal suppression, but its slightly deprecated. If you are on a Linux system, your in luck:
simply delete the folder named lanms from this directory, and pip install lanms-nova, should work okay.
If that doesnt work, you can find the original east text detector repo by argman, repo name EAST.
the lanms folder there will have an original makefile set up for a linux user.

If you are on a mac like myself, you're going to need to troubleshoot that make process until it works, or just
replace the lanms implementation in the code with vanilla nms. 

