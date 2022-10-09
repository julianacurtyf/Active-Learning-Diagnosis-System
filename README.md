# An Active Learning-based Medical Diagnosis System

We developed an AI system to help clinicians in their daily practice. They could consult the system to get an immediate opinion and diminish waiting times in triage services since this task could be carried out with minimal human interaction. Our method relies on Machine Learning techniques, more precisely on Active Learning and Neural Networks classifiers. 

The dataset used can be found with more information at: https://www.kaggle.com/itachi9604/disease-symptom-description-dataset?select=dataset.csv. Also, the three csv files in this repository, namely dataset.csv, target.csv and Symptom-severity.csv contain all the data preprocessed necessary to run the program.


## Setup

In this project, an Active Learning model was developed, and it works as a medical diagnosis model with a neural network classifier. This study was conducted using Python (4.2.5) and Active Learning through the modAL library (0.4.1). 

To execute this program, some libraries are required, they can be found in requirements.txt. The versions are indicated due to the fact they do not have any incompatibility between themself. 

If you are trying to execute the version .py, we recommend that you put the dataset at the same directory as the code. The following command should be run in a terminal (Anaconda prompt or Python, itself):

```sh
$ pip install -r requirements.txt
```

If you execute the version .ipynb, we also recommend that you maintain the datasets in the same folder as your code. In case the datasets and code are not in the same folder, the path can be changed at cell 5. In order to run the program, is not necessary to install any package separately, those commands are already on the first cells. 


## Cite this paper

In the case you use these scripts, cite the following manuscript:

Pinto, C., Faria, J., Macedo, L. (2022). An Active Learning-Based Medical Diagnosis System. In: Marreiros, G., Martins, B., Paiva, A., Ribeiro, B., Sardinha, A. (eds) Progress in Artificial Intelligence. EPIA 2022. Lecture Notes in Computer Science(), vol 13566. Springer, Cham. https://doi.org/10.1007/978-3-031-16474-3_18


Bibtex:
@InProceedings{10.1007/978-3-031-16474-3_18,
author="Pinto, Catarina
and Faria, Juliana
and Macedo, Luis",
editor="Marreiros, Goreti
and Martins, Bruno
and Paiva, Ana
and Ribeiro, Bernardete
and Sardinha, Alberto",
title="An Active Learning-Based Medical Diagnosis System",
booktitle="Progress in Artificial Intelligence",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="207--218",
abstract="Every year thousands of people get their diagnoses wrongly, and several patients have their health conditions aggravated due to misdiagnosis. This problem is even more challenging when the list of possible diseases is long, as in a general medicine speciality. The development of Artificial Intelligence (AI) medical diagnosis systems could prevent misdiagnosis when clinicians are in doubt. We developed an AI system to help clinicians in their daily practice. They could consult the system to get an immediate opinion and diminish waiting times in triage services since this task could be carried out with minimal human interaction. Our method relies on Machine Learning techniques, more precisely on Active Learning and Neural Networks classifiers. To train this model, we used a data set that relates symptoms to several diseases. We compared our models with other models from the literature, and our results show that it is possible to achieve even better performance with much less data, mainly because of the contribution of the Active Learning component.",
isbn="978-3-031-16474-3"
}
