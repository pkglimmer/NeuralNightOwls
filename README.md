## Scripts

*train_recon.py*: train the autoencoder for reconstruction, only for testing the capacity of the autoencoder;
*train_dagmm.py*: DAGMM training script, with energy calculations for each epoch included;
*inference.py*: calculate the reconstruction loss, energy and covariance diagonal for the samples, save to a table;
*train_cls.py*: perform classification using i. the pretrained encoder, ii. selective sampling based on the stats calculated in the inference step.

*train_classification.py*: train the training datasets with classification, log both training result and validation result 
*train_segmentation.py*: train the training datasets with segmentation, log both training result and validation result 


