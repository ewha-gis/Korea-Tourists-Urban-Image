# Training inception-v3 model on korea_tourist_photo_dataset

- inceptionv3_retrain.py 
transfer learning code

python inceptionv3_retrain.py --image_dir [image_dir] --how_many_training_steps [training_steps(ex.10000)] --learning_rate [learning_rate(ex. 0.001)] --train_batch_size [batch_size(ex.150)]

Traing steps, learning rate and train batch size depend on which dataset you want to train

- classify_multigpu.py 
classify images by using multi-gpu
