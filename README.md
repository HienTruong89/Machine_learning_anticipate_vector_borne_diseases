# Project Abstract
This repository provides Machine Learning (ML) strategies for predicting disease stages using time-series data and temporal image representations, leveraging both early and late fusion techniques.
Methodology:
A synthetic time-series dataset was generated based on SIR-like epidemiological compartmental models. Two classification strategies—early fusion and late fusion—were applied to time-series and temporal image data to predict disease stages (e.g., low, medium, and high risk of outbreaks). Time-series data were transformed into images using Standardised Euclidean Distance Recurrence Plots, Gramian Angular Fields (Summation & Difference), and Markov Transition Fields. These encoded images were combined into RGB formats to enrich feature representation. Dominant features from temporal images were extracted using pre-trained ImageNet models. Finally, various ML algorithms were applied to either the extracted image features or the raw time-series data to anticipate stages of vector-borne diseases.

# 1. Introduction
Vector-borne diseases (e.g. West Nile Virus), transmitted to human by infected mosquitoes, ticks, and fleas, pose significantly public health concerns, especially with the increasing incidence linked to climate change.
This project explores the potential of ML approaches to anticipate vector-borne disease outbreaks. 
While many studies over the past decades have focused on predicting epidemic or early warning signals of tipping points using traditional time-series data, our approach shifts to a novel image-based time-series representation. 
Leveraging the advancements in Artificial Intelligence for image recognition and detection, this project aims to enhance the prediction of impending disease outbreaks. 

# 2. Image-Based Time-Series Representation
Three different encoding methods, including the recurrent plot (PR), Gramian angular field (GAF), and Markov transition field (MTF), to generate time-series image-based representations from scalar time-series data were investigated. 
For n observations, the time series x(t) is written below. 
                                                                   x(t) = { x1, x2, x3, ….,xn}

The main idea was that these encoding methods transform time-series data into high-dimensional image representations, facilitating the search for temporal correlations and transitions between data points. 

# 2.1. Encoding time-series with Recurrent Plot method

The recurrent plot (RP) method is a transformative technique where the elements in the matrix represent the actual distances between points in the time series (Marwan et al., 2007). This study constructed a recurrence plot image (ReIM) using the pairwise standardized Euclidean distance between elements of a time series x(t), which takes an account for the correspondent variance across the dimension of the original data (SciPy, 2008-2024). The condensed distance matrix was then converted into a square matrix form, generating a recurrent plot image (ReIM).

                                                                  dij= √(∑(x_i-x_j)^2/V_i)
                                                                   
where dij is the distance matrix at a certain i row and j column, V_i is the dimensional variance of the time-series data for i row.

                                                                   Recurrent Plot = RP [i, j] = dij

# 2.2. Encoding time-series with Gramian Angular Field method
Gramian Angular Field (GAF) is a method that converts time series data into an image based on the Gramian matrix (G = XTX). Before the transformation, the time series data points are normalized in the range of -1 and 1 and then converted into angular radians (∅) using inverse cosine function. There are two primary methods: summation and difference. The Gramian Angular Summation Field (GASF) calculates cosine-based summation between adjacent angular time steps, while the Gramian Angular Difference Field (GADF) focuses on differences in sine-based radians. Although GASF and GADF are opposite approaches, they both preserve temporal dependencies and emphasize both the amplitude and phase information in the original time series data (Dias, Dias, et al., 2020; Dias, Pinto, et al., 2020). 

                                                                 〖GASF〗_(i,j)=cosine⁡〖 (∅_i+ ∅_j )〗
                                                                 〖GADF〗_(i,j)=s⁡ine(∅_i- ∅_j )

where ∅_i=arccos⁡〖(x_i 〗) and x_i is normalized to [ -1, 1]. 

# 2.3. Encoding time-series with Markow Transition Field method 
Markov Transition Field (MTF) is another encoding method that transforms a time series into an image using a Markov model. The time series is initially normalized to a range between 0 and 1 and then divided into quartiles or bins based on the normalized values. A Markov model sequentially encodes these bins into state transitions, computing the probability of transitioning from one state to another. The MTF matrix element at row i and column j is then calculated based on this probability transition matrix (Dias, Dias, et al., 2020; Dias, Pinto, et al., 2020).   

                                                                 MTF [i,j] = P (st = j |st-1 = i)
where MTF [i,j] represents the probability of transitioning from state i to state j. 

Overall, the MTF method illustrates the relationship between two arbitrary points in the time series and how frequently these transitions occur. The MTF is sensitive to the choice of bin levels, which can affect the granularity of the original time series. 

# 3. Early fusion approach for time-series images
The current study employed a early fusion strategy that integrates different encoding methods Recurrent Plot (RP), Gramian Angular Summation Field (GASF), Gramian Angular Difference Field (GADF), and Markov Transition Field (MTF) to capture complementary information embedded in time-series image representations.
This approach, adapted from the work of Dias, Pinto, et al., (2020), involves assigning each encoded image representation to one of the three color channels of an RGB image. While this technique has been utilized in the fields such as phenology (Dias, Pinto, et al., 2020; Faria et al., 2016), it has not been previously applied for predicition of vector-borne diseases using time-series. 

Mosquito vector-borne diseases typically involve five key compartments: mosquitoes infected (MI), birds infected (BI), birds recovered (BR), quinines infected (QI), and humans infected (HI). This study focuses on three primary compartments: MI, BI, and BR. 
The time-series data for each compartment were encoded using different encoding methods, and these encoded images were fused into an RGB image. The orders in which the compartment are assgined to the RGB channels significantly influences the resulting image representations.  

This research specifically explores the fusion of three recurrent plots into an RGB image, as well as a combination of GASF, GADF, and MTF into an RGB image. For all early fusion, the MI, BI, and BR compartments were mapped to the blue, green, and red channels, respectively.  

![image](https://github.com/user-attachments/assets/400d8828-c45b-475e-8c57-c0d3fd3519b4)

Figure 1. Fusion of the three recurrent plots (a) and a combo of GASF, GADG, and MTF (b) into an RGB image

# 4. Late fusion approach
Each variable of three image channels (RGB image) was modelled separatedly using ML and then fused decision to make a final classification based on majority voting.  

# 5. Machine learning workflow 
The current study proposes achine learning workflow consisting of data collection, data processing, feature extraction, model training, and evaluation. 
Key innovation induces employing diverse encoding methods to transform scalar time-series data into image-based representations and leveraging transfer learning from feature extractors to enhance the prediction of mosquito vector-borne outbreaks. 


![image](https://github.com/user-attachments/assets/dead9dbb-9f6b-4d17-8b82-f538de266aae)

Figure 2: Encoding scalar time series data using the recurrent plot method and a fusion of three recurrent plots into either a recurrent plot image (ReIM) or channels of a red-green-blue image (RGB), and fit into different machine learning models for prediction of tipping point or stable events.

# 5.1. Feature extractors
This study utilized various ImageNet architectures as feature extractors for the time-series images related to vector-borne diseases. ImageNet is a well-known convolutional neural networks trained on a vast visual dataset comprising millions of images, primarily used for image recognition tasks (Lukas et al., 2022). 
In this study, ImageNet function acts as a feature extractor, with its pre-trained networks used, while keeping the flattened and classifying layers frozen. The feature extractor generates flattened deep features, which are then input into machine learning classifiers (Ebrahim et al., 2019). 

We employed seven common ImageNet architectures in the current study, namely, ViT32, VGG16, ResNet50, ResNet101, ResNet152, Xception, and Efficient_NetB5. These architectures differ in their neural network designs and transfer learning algorithms (Morid et al., 2020). 
Utilizing these pre-trained models can offer different approaches for training can reduce training time and improve performance compared to building neural networks from scratch. For more information on ImageNet and its advantages, and disadvantages, please refer to the Keras Applications on the Keras Applications website. 

# 5.2. Machine learning 
Six different machine learning argorithms: Logistic Regression (LR), Random Forest (RF), Extreme Gradient Boosting (XGB), K-Nearest Neighbor (KNN), Support Vector Machine (SVM) with both linear and radial basic functions (rbf), Multi-layer Perceptron (MLP) were employed on image data to classify three scenarios of vector-borne diseases. 
On constrast, MOMENT, Lag_Llama, and LSTM are implemented on time-series data 
The choice of these ML techniques is based on practical experience and general knowledge of their effectiveness. 

# 5.3. Deep learning 
1D-CNN-LSTM was used for both time-series and image data. 

# 5.4. Fine-tuning 
Fine-tuning is used in this study as an optimization technique to enhance the performance of ImageNet’s final layers when adapted as classifiers for vector-borne diseases. During feature extraction, the flattened and connected layers of ImageNet were removed and replaced by either new fully connected layers or by traditional machine learning methods serving as classifiers (Sarkar et al., 2018). Hyperparameters for these layers were selected through random screening based on empirical knowledge. Additionally, the GridSearch function in the Python Scikit-Learn package was employed to identify the optimal parameters for traditional machine learning classifiers (Morid et al., 2020). 

# 6. Tools and Environment 
Python 3.11 on WUR’s High-Performance Computing Cluster (Anunna) for data processing, modeling, and validation.

# 7. Python codes 
The Python codes provided in this repository includes time-series image generators using aforementioned encoding methods mentioned, as well as the deployment of differnt machine learning agorithms to anticipate vector-borne diseases. 
- Image generators
- Machine learning models
- Fine-tuning enhancing models




                                      





