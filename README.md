# Adversarial-Attacks-on-Aerial-Scene-Classification
This repository contains the code and results for the project titled **Adversarial Attacks on Aerial Scene Classification** that I accomplished during my summer research internship at **VIGIL Lab**, **IIT Hyderabad** under the guidance of Prof. C. Krishna Mohan

## Our Contributions
- We implemented white-box adversarial attacks using 5 different neural network
architectures on different aerial scene classification datasets under two different
configurations: eps=0.0005 and eps=1.0.
- We tabulated our results and inferred the best-attack on each dataset using a
particular network architecture. Confusion Matrix were also plotted for evaluation
purposes.
- We implemented the Transferable Sparse Adversarial Attack(TSAA) and
deployed it on our dataset to infer the transferability of this black-box attack on
our datasets using different neural network architectures.
- We implemented state-of-the-art black box attacks on our datasets using
Adversarial Attack libraries.

## Data
The datasets that were used for the study were:
- **NWPU-RESIC45** : RESISC45 dataset is a publicly available benchmark for Remote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class. 
-  **UC Merced Land Use Dataset** : UC Merced is a 21-class land use remote sensing image dataset, with 100 images per class. The dataset contains 2100 images which were manually extracted from large images from the USGS National Map Urban Area Imagery collection for various urban areas around the country. The pixel resolution of this public domain imagery is 0.3 m.

## Neural Network Architectures Used
- AlexNet
- ResNet50
- ResNet101
- MobileNet
- DenseNet-121

## Adversarial Attack Algorithms
- **White-Box Adversarial Attack** :-

      1. Gradient-Based Attacks: Gradient Attack, Gradient Sign Attack (FGSM), Iterative Gradient Attack, Iterative Gradient Sign Attack, DeepFool L2 Attack, DeepFool L∞ Attack, L-BFGS Attack, SLSQP Attack, Jacobian-Based Saliency Map Attack 
      2. Score-Based Attacks: Single Pixel Attack, Local Search Attack, Approximate L-BFGS Attack 
      3. Decision-Based Attacks: Boundary Attack, Pointwise Attack, Additive Uniform Noise Attack, Additive Gaussian Noise Attack, Salt and Pepper Noise Attack, Contrast Reduction Attack, Gaussian Blur Attack, Precomputed Images Attack

- **Transferable Sparse Adversarial Attack** : TSAA is a sparse adversarial attack based on the L0 norm constraint, which can succeed by only modifying a few pixels of an image. Prior sparse attack methods achieve a low transferability under the black-box protocol because they methods rely on the target model’s accurate gradient information or its approximation, causing the generated adversarial examples overfitting the target model. TSAA is a trainable generator-based architecture which tends to alleviate the overfitting issue by learning to translate a benign image into an adversarial example and thus efficiently craft transferable sparse adversarial examples	
 
<p align="center">
  <img width="500" height="200" src="https://user-images.githubusercontent.com/68850685/129438113-7f698587-ccd0-4c26-bd34-545446c1d792.png">
</p>

## Experiment Details
- Deep Learning Framework: Pytorch
- GPU: Tesla M60(batch size=16)
- Optimizer: Adam Optimiser(learning rate=0.01, weight decay rate= 0.001)
- Loss Function: Cross Entropy Function
- Epochs: 100
- Train Set : Test Set :: 80:20
- Epsilon: 0.0005; 1.0
- Evaluation Metrics: Accuracy, Confusion Matrix

## Results

A.	White-Box Attacks

1)	NWPU-RESIC45

•	Epsilon: 0.0005

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438238-814f3941-365c-409e-9874-f60394b4d7b1.png">
</p>























Confusion Matrix for Virtual Adversarial Attack on AlexNet



•	Epsilon: 1.0























Confusion Matrix for Newton Fool Attack on DenseNet121




2)	UC Merced Land Use Dataset

•	Epsilon: 0.0005





















Confusion Matrix for PGD Attack on MobileNetV2



•	Epsilon: 1.0




















                           Confusion Matrix for L2-Projected Gradient Descent Attack on AlexNet




B.	Transferable Sparse Adversarial Attack (Black-Box Adversarial Attack)

1)	NWPU-RESIC45

•	Epsilon: 0.0005




















                                                Confusion Matrix for ResNet101 Generator Attack on ResNet50




•	Epsilon: 1.0




















                                   Confusion Matrix for ResNet101 Generator Attack on ResNet50
 


2)	UC Merced Land Use Dataset

•	Epsilon: 0.0005


















                                  Confusion Matrix for MobileNetV2 Generator Attack on ResNet101
 



•	Epsilon: 1.0



















                      Confusion Matrix for ResNet50 Generator Attack on ResNet50

