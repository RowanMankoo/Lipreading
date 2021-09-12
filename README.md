# Lipreading
This repository contains all the code for the summers masters project
## File Structure

**Data**: Contains custom made dataset class with accompanying data loader for the LRS2 dataset

**Model Modules**: Contains all modules used to build the model (3D CNN, ResNet-18, encoder-decoder)

**Final Model**: Contains the final model class, load this to get the model used in the paper

**Training**: Contains the curriculum and full training loops used

**Testing**: Contains file to calculate test WER/CER and also code to produce saliency maps plots and loss plots

**Utils**: Contains miscellaneous processing functions to be used on model predictions and text

To use the model only the Data,Model Modules, and Final Model files are required. However, to train and test the model, all files are required.
