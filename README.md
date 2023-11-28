# CS337 Project - Disagreement Learning

Team SaNGiT

- Ashwin Goyal
- Aditya Nemiwal
- Atharva Abhijit Tambat
- Divyansh Singhal

## Overview

We demonstrated that using Disagreement Learning can improve the diversity of the various models trained and hence encourages better generalization for the ensemble. This also ensures that the model does not make costly mistakes when it is unsure and decreases the confidence on predicting images it isn't supposed to predict. Disagreement learning also helps counter simplicity bias

## Our Tasks

1. Hypothesis 1 - We want to demonstrate that using the disagreement loss causes the model to learner harder features compared to just the cross-entropy loss. This makes the model more generalizable.
2. Hypothesis 2 - We wish to analyse how the number of classes in a classification task affect the ability of the disagreement loss to improve upon standard ensembling
3. Hypothesis 3 - We wish to show that the accuracy improves with the number of models in the DBAT loss. I.E ensembling over more models, trained to disagree is better for accuracy.
4. We demostrated the results of the paper with smaller models and analysed how the results changed from what they reported



## How to run code
For Hypothesis 1 - Run domino.ipynb
For Hypothesis 3 - Added script test_num_classes.ipynb







## References

[Docs](https://docs.google.com/document/d/1haGnOLEIGB9FBDRNcDlD97-WADgQ26CzZdLJveqQoCI/edit)
[Agree-to-Disagree](https://openreview.net/pdf?id=K7CbYQbyYhY)
[Model](https://drive.google.com/file/d/18t0W_N0MLVU26jibg0eeOXUPKD1TaP7A/view)

[slides](https://docs.google.com/presentation/d/1Me-VbeTc64zNqFGPmz2Rw9wZ82AhXt5pix_1RYk4w5c/edit#slide=id.p)

[Overleaf](https://www.overleaf.com/project/6564447a9351db5caa642559)
