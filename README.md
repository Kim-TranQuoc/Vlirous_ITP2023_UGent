# Vlirous_ITP2023_UGent
This is a part of works of Vlir-ous International Training Progamme (ITP) 2023 at Ghent university with project title of "Machine learning for climate change mitigation in buildings". This work includes the following files:
- Data_Analysis.py: Analyzing the simulation data with statistical features, histogram & density plots, data visualization with parallel coordinates plots, and variable correlation coefficients;
- Model_1.py: A simple ANN model with 2 phases (train and test);
- Model_2.py: A simple ANN model with 3 phases (train, validation and test) and early stopping codition (based on val_loss);
- Model_3.py: A simple ANN model with 3 phases (train, validation and test) and custom early stopping codition (based on val_loss and loss_relation);
- Predict_Model.py: Generating predictions via a saved model;
- Model_Agnostic_Analysis.py: Analyzing the trained model with partial dependence plots and feature importance,
- Metaheuristic_Optimization.py: Optimizing the problem with ANN surrogate model and metaheursitic algorithm DE.

# Contributor
- Kim Q. Tran

# References
- Model agnostic analysis: [https://christophm.github.io/interpretable-ml-book/pdp.html](https://christophm.github.io/interpretable-ml-book/)
- Partial dependence plots: Friedman, Jerome H. “Greedy function approximation: A gradient boosting machine.” Annals of statistics (2001): 1189-1232
- Feature importance: Greenwell, Brandon M., Bradley C. Boehmke, and Andrew J. McCarthy. “A simple and effective model-based variable importance measure.” arXiv preprint arXiv:1805.04755 (2018)
