# Brain age model

In this directory, I made available the code used to train the age prediction model of the article "Longitudinal analysis of AI predicted brain age in amnestic and non-amnestic sporadic early-onset Alzheimer's disease".

The code is divided into five files : 
- *callbacks.py* - which contains the custom callbacks functions ;
- *data_agumentation.py* - which contains data augmentation functions on three dimensional images;
- *generators.py* - which contains the generators able to create our batch of MRI images;
- *regression.py* - which contains the architecture of the model used;
- *brain_age.ipynb* - the notebook that uses each of the above functions to train the age prediction model. 
