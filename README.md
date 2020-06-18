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

For example, if there exists 15 feature interactions, in the print log, you may get the architecture parameter after search stage as follows
**log avg auc all weights for(epoch:None,wt_config:init:0.6+-0(seed=2017) l1:0 l2:0 act:none) 0.01,0.0,0.0,0.0,0.04,0.53,0.32,0.3,0.0,0.0,0.12,0.3,0.0,0.0,0.12,**

Then in the retrain stage, you could set "comb_mask" as follows
**comb_mask = \[1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1\]**

The third feature interactions are similar to that.
