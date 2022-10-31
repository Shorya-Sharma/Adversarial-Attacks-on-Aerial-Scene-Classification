# Adversarial-Attacks-on-Aerial-Scene-Classification
This repository contains the code and results for the project titled **Adversarial Attacks on Aerial Scene Classification** that I accomplished during my summer research internship at **Pattern Recognition Lab**, **University of Erlangen-Nuremberg** under the guidance of Prof. Andreas Maier

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

### A.	White-Box Attacks

#### 1) NWPU-RESIC45

- Epsilon: 0.0005

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438238-814f3941-365c-409e-9874-f60394b4d7b1.png">
</p>


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438274-066a3a80-8bf7-416c-b71e-1a792cc59427.png">
</p>
<div align="center">
  Confusion Matrix for Virtual Adversarial Attack on AlexNet
</div>


- Epsilon: 1.0

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438475-7f39c0bb-c350-499f-866a-73d6d6174d41.png">
</p>


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438483-214597e9-5d3f-47f4-bda4-fafe1cbbfd87.png">
</p>
<div align="center">
  Confusion Matrix for Newton Fool Attack on DenseNet121
</div>


#### 2) UC Merced Land Use Dataset

- Epsilon: 0.0005

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438564-c5d2e809-53ad-4312-a049-dd04c71e742e.png">


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438581-c438affc-f453-4036-b601-de3b36f6cd20.png">
</p>
<div align="center">
  Confusion Matrix for PGD Attack on MobileNet
</div>


- Epsilon: 1.0

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438599-d78519c1-f0a3-4be3-804a-35a938be9573.png">
</p>

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438617-656625a4-58ef-4277-84fc-2f71ce66b39a.png">
</p>
<div align="center">
  Confusion Matrix for L2-Projected Gradient Descent Attack on AlexNet
</div>


### B. Transferable Sparse Adversarial Attack

#### 1) NWPU-RESIC45

- Epsilon: 0.0005

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438662-7584732f-2075-4f95-a821-b895d5c9b91d.png">
</p>

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438674-25a7853e-94c0-430d-9faa-b7231efa90ac.png">
</p>
<div align="center">
  Confusion Matrix for ResNet101 Generator Attack on ResNet50
</div>


- Epsilon: 1.0

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129439085-4f90bfb0-1205-49cd-9897-3b293108cc81.png">
</p>

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129439097-b575991a-ac5a-43f8-befb-6ead6730bae1.png">
</p>
<div align="center">
  Confusion Matrix for ResNet101 Generator Attack on ResNet50
</div>


#### 2) UC Merced Land Use Dataset

- Epsilon: 0.0005

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438716-27c7de4c-6000-4da6-8589-1efce9bee919.png">

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438720-4ee01b53-19ab-4e54-86b9-24f0e0db4db1.png">
</p>
<div align="center">
   Confusion Matrix for MobileNetV2 Generator Attack on ResNet101
</div>


- Epsilon: 1.0

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129438732-d2a456dd-e3fe-4f3b-9aeb-7be4bf293f99.png">
</p>


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/68850685/129439045-26cd9a94-fbbd-4449-9a01-07d2c9201949.png">
</p>
<div align="center">
   Confusion Matrix for ResNet50 Generator Attack on ResNet50
</div>


## Observations
Following Observations were made on the basis of the experiments conducted in the above study:
- On NWPU-RESIC45, the highest drop in accuracy was observed when Virtual Adversarial Attack (75.23%) and Newton Fool Attack (44.67%) were deployed under eps= 0.0005 and eps=1.0 respectively. to create adversarial examples on the AlexNet and DenseNet respectively.
- When TSAA was executed on NWPU dataset, highest fooling rate was 9.30% when the generator was trained on ResNet101 architecture and the adversarial examples were created using the ResNet50 model under eps=0.0005 and 3.3% when the generator was trained on ResNet101 architecture and the adversarial examples were created using the ResNet50 model under eps=1.0
- On UC Merced Dataset, the highest drop in accuracy was observed when L2 Projected Gradient Descent Attack (4.01%) and PGD Attack (98.24%) were deployed under eps= 0.0005 and eps=1.0 respectively. to create adversarial examples on the AlexNet and MobileNetV2 respectively.
- When TSAA was executed on UC Merced dataset, highest fooling rate was 15.27% when the generator was trained on MobileNetV2 architecture and the adversarial examples were created using the ResNet101 model under eps=0.0005 and 6.44% when the generator was trained on ResNet50 architecture and the adversarial examples were created using the ResNet50 model under eps=1.0



