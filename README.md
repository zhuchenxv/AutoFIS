# AutoFIS
AutoFIS: Automatic  Feature Interaction Selection in Factorization  Models for  Click-Through Rate Prediction

This is the tensorflow implementation of AutoFIS in the following paper:

Bin Liu, Chenxu Zhu, Guilin Li, Weinan Zhang, Jincai Lai, Ruiming Tang, Xiuqiang He, Zhenguo Li, Yong Yu. AutoFIS: Automatic Feature Interaction Selection in Factorization Models for Click-Through Rate Prediction. KDD 2020.  


# How to use
init.py contain some parameter setting and other in main file.
Dataset download link and how to use could be seen in dataset.txt.

Our model AutoFIS has two stages: search stage and retrain stage. You could run model and get important interactions. According to the important interactions, you change the **comb_mask** and **comb_mask_third** in the main file. In the search stage, for architecture parameters, you have to use Grda Optimizer. In the retrain stage, you have to apply Adam Optimizer.

run tf_main_autofm.py could apply AutoFIS to FM and it is the main file of AutoFM.

run tf_main_autodeepfm.py could apply AutoFIS to DeepFM and it is the main file of AutoDeepFM.
