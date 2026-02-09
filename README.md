# Unsupervised ensemble pruning technique for Isolation Forest

## Abstract
This study focuses on the field of Anomaly Detection (AD) which aims to identify
unusual behaviors relative to a specific definition of normality in a specific context.
Anomalies are rare events that can have significant implications, in terms of safety,
security and operational efficiency, in various application domains, such as medicine,
finance, and industry.
A widely used algorithm in AD, adopted in this study, is the Isolation Forest (IF),
a standard choice for this task due to its lightweight nature and unsupervised
formulation: anomalies, which are few and structurally different, are easier to
isolate than normal points within the recursive partitioning structure of trees. The
strength of this model lies in its randomized recursive partitioning process. By
randomly and recursively selecting both features and split values, each tree generates
a distinct isolation structure. When aggregated into an ensemble, this diversity
makes the model not only efficient but also robust. However, this randomness may
also introduce redundancy and noise: some trees may be weakly informative, carry
an insignificant structure, or even degrade the overall anomaly score.
This observation motivates the scope of this study: ensemble pruning of the Isolation
Forest. The central hypothesis is that harmful or non-informative trees can be
identified and removed, thereby reducing the size and computational complexity
of the model and potentially improving the quality of the anomaly scores. The
challenge is that in unsupervised anomaly detection, true labels are most often un-
available, making traditional pruning strategies — most of which rely on supervised
evaluation — inapplicable.
Furthermore, many of the Artificial Intelligence (AI) models used for Anomaly
Detection, including the Isolation Forest, are complex and cannot be easily in-
terpretable, which is essential in safety-critical applications. In fact, the lack of
interpretability can lead to a lack of trust in the system and, in some cases, to
catastrophic consequences. Therefore, it is important to provide explanations for
the decisions made by AI systems. For these reasons, the eXplainable Artificial
Intelligence (XAI) or Interpretable Machine Learning field is a recent research
area that aims to provide tools and techniques to make AI systems more human-
understandable, allowing them to gain insight into the decision-making process.
Then, to address such challenge, experiments were conducted in order to develop
a novel unsupervised pruning strategy for the Isolation Forest by exploiting two
main research directions:
1. Investigating the potential of interpretability techniques like DIFFI as ex-
planatory tools and also as mechanisms for model optimization. The basic
idea is to leverage the feature importance scores provided by DIFFI to identify
and prune trees that contribute to overall performance of the ensemble.
2. Investigating the intrinsic properties of the Isolation Forest, focusing in par-
ticular on the depth distribution to identify trees that may not be informative
about ensemble performance.
DIFFI, an explainable technique specifically tailored for Isolation Forest, is an
algorithm that returns both global and local feature importance scores, which are
not only useful for providing explanations of the model decisions to end-users, but
also allow developers to better understand the global functioning of the ensemble.
The experiments carried out span various synthetic and real-world datasets, allowing
us to assess the effectiveness of our proposed pruning strategies under different data
distributions and anomaly patterns.
The results demonstrate that the DIFFI-based strategy has the potential to improve
Isolation Forest performance; however, its effectiveness is highly dataset-dependent
and not universally guaranteed. On the other hand, the Depth-based strategy shows
more consistent improvements across different datasets, indicating that leveraging
the depth property of the Isolation Forest, i.e., the anomalies are likely to be isolated
at shallower levels while the normal instances are likely to be isolated at deeper
levels, it is a more reliable approach for ensemble pruning in unsupervised anomaly
detection tasks. Indeed, the depth-based Separation Ratio method, where trees
are evaluated based on the ratio of the average depth of inliers to that of outliers,
demonstrates significant improvements in both F1-Score and Average Precision
across a wide range of datasets.
Then further analysis is conducted to assess the impact of the number of trees in
the original Isolation Forest on the effectiveness of pruning. The results suggest
that identifying an optimal number of trees is crucial to maximizing the benefits of
pruning, in order to further improve the performance and efficiency of the Isolation
Forest model.

## Further reading
Read `thesis.pdf` for a complete description of the thesis work.
Read `thesis_presentation.pdf` for a brief description of the thesis work.
