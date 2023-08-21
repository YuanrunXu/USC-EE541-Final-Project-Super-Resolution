# USC-EE541-Final-Project
CV Super Resolution

1.	Project Overview
The project contains 4 folders:
	Bicubic: code of Bicubic interpolation
	Data: contains DIV2K dataset
	SRCNN: code and model of SRCNN
	SRGAN: code and model of SRGAN
 
1.1	Bicubic
 
	Bibubic.py: apply Bibubic interpolation by using cv2.INTER_CUBIC to the loading data.
	dataLoading.py: load train, validation, and test dataset from the Data folder
	evaluate.py: methods of calculating SNR, PSNR, and SSIM.
1.2	Data
 
Data folder contains a folder named DIV2K. DIV2K contains the train, validation, and test dataset. 
Images are directly in HR folders: DIV2K_test_HR, DIV2K_train_HR, and DIV2K_valid_HR:
 
However, images in LR folders are under a sub-folder named X2:
 
	dataOverview.py: shows the overview of the dataset:
	dataPrepare.py: split and rename the images:
1.3	SRCNN
 
	saved_model folder contains the trained model.
	dataLoading.py: dataLoading.py: load train, validation, and test dataset from the Data folder
	evaluate.py: methods of calculating SNR, PSNR, and SSIM; and method of calculating these metrics during training.
	evaluate_test.py: above during testing; call image_show() to show the output images.
	Image_show.py: method to show the output images
	model.py: define the class SRCNN
	plot.py: plot the training curves
	test.py: test the model after training
	train.py: train the model

1.4	SRGAN
The structure of SRGAN folder is the same as SRCNN.
 

2.	Deployment
	Put the split DIV2K dataset into the Data folder:
Train:valid:test 5:3:1
 
	Bicubic interpolation:
Run bicubic.py
	SRCNN
Training: run train.py; if the saved_model folder does not exist, it will be created automatically before training; the model will be saved to the folder. Training curves will be saved to SRCNN folder after training.
Testing: run test.py. LR, HR, and output images will be shown after testing.
	SRGAN: the same as SRCNN
![image](https://github.com/YuanrunXu/USC-EE541-Final-Project-Super-Resolution-/assets/43257371/e059dfb1-7339-4e4d-8ec7-214d0314af62)

