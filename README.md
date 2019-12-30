# EE838A_HW1
Homework 1, Advanced Image Restoration and Quality Enhancement, EE, KAIST, Fall 2018

1. Training
		- Open command window or terminal in 'source' folder
		- Type: python main.py --mode=train

		+ All logs of the training process are recorded in the file "../logs/logs_train.txt"
		+ After training, the weights of the network are saved in '../model' folder

2. Validation
		- If you want to validate my best model, copy all files from '../report/model' to '../model', otherwise skip this step
		- Open command window or terminal in 'source' folder
		- Type: python main.py --mode=valid

		+ All logs of the validation process are recorded in the file "../logs/logs_valid.txt"
		+ All generated Super-Resolution (SR) images are saved in "../report/valid_sr_gen"
		+ All PSNR and SSIM values are written in the file "../report/scores_valid.txt"

3. Test
		- If you want to test my best model, copy all file in "../report/model" to "../model", otherwise skip this step
		- Copy all the testing High Resolution (HR) images into folder "../data/test/HR"
		- Copy all the testing Low Resolution (LR) images into folder "../data/test/LR"
		- Open command window or terminal in 'source' folder
		- Type: python main.py --mode=test

		+ All logs of the testing process are recorded in the file "../logs/logs_test.txt"
		+ All generated SR images are saved in "../report/test_sr_gen"
		+ All PSNR and SSIM values are written in the file "../report/scores_test.txt"

4. Report
	The report of homework 1 is written in the file "../report/EE838A_HW1_Report_DinhVu_20184187.docx"
