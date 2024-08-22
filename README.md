# AI_pinelines_anticipate_vector_borne_diseases

# 1. Introduction
Vector-borne diseases are illness transmitted to human by infected mosquitoes, ticks, and fleas. These diseases are highly concerned by public health due to their dynamic increases corresponding to climate change.
The current project investigates the potential of different AI pinelines to anticipate vector borne diseases. Over the past decades, many studies have been conducted to predict the epidemic of these deseases, but mostly using collected time-series datapoints. In the current project, we focus on anticipating vector-borne deseases based on image-based time-series representation. The rapid development of AI generations on image recognition and detection could help to predict the outbreaks. 

# 2. Image-based time-series representation methodologies
Three different encoding methods, including the recurrent plot (PR), Gramian angular field (GAF), and Markov transition field (MTF), to generate time-series image-based representations from scalar time-series data were investigated. 
For n observations, the time series x(t) is written below. 
                                                                   x(t) = { x1, x2, x3, ….,xn}

The main idea was that these encoding methods transform time-series data into high-dimensional image representations, facilitating the search for temporal correlations and transitions between data points. 

# 2.1. Encoding time-series with recurrent plot method

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
The current study employed a fusion of different encoding methods (RP, GASF, GADF, and MTF) to unravel complementary information embedded in time-series image representations and among the classifiers used (Dias, Pinto, et al., 2020). The early fusion method was implemented by assigning each encoded image representation to one of the three color channels of an RGB image. While this approach has been applied in other scientific research fields such as phenology (Dias, Pinto, et al., 2020; Faria et al., 2016), it has not been previously reported for time-series vector-borne diseases. 
Mosquito vector-borne diseases typically consist of five compartments: mosquitoes infected (MI), birds infected (BI), birds recovered (BR), quinines infected (QI), and humans infected (HI). This study focused on three primary compartments: MI, BI, and BR. The time-series data of each of the three components were transformed using different encoding methods, and each encoded image was then fused into an RGB image. The order of fusion of each compartment plays a crucial role in generating different RGB images. 
This research specifically examines the fusion of three recurrent plots into an RGB image and a combination of GASF, GADF, and MTF into an RGB image. For the combination fusion, the MI, BI, and BR compartments were assigned to the blue, green, and red channels, respectively.  

![image](https://github.com/user-attachments/assets/400d8828-c45b-475e-8c57-c0d3fd3519b4)

Figure 2. Fusion of the three recurrent plots (a) and a combo of GASF, GADG, and MTF (b) into an RGB image

# 4. AI pinelines 
AI pineline refers to structured workflows utilized to develop, deploy and maintain machine learning models. It consists of various processes including data collection, data processing, data extraction , model training, model evaluation, model deployment, monitoring and maintenance, and feedback loop. Integration of AI pinelines help ensure AI solutions manageble and that models are robust, scalable and maintainable. 

The current study explores various AI pipelines aimed at anticipating stages of vector-borne diseases. The key innovations induce employing diverse encoding methods to transform scalar time-series data into image-based representations and leveraging transfer learning from feature extractors to enhance the prediction of mosquito vector-borne outbreaks. 


![image](https://github.com/user-attachments/assets/dead9dbb-9f6b-4d17-8b82-f538de266aae)

Figure 1: Encoding scalar time series data using the recurrent plot method and a fusion of three recurrent plots into either a recurrent plot image (ReIM) or channels of a red-green-blue image (RGB), and fit into different machine learning models for prediction of tipping point or stable events.

# 4.1. Feature extractors
In this study, we utilized different ImageNet architectures as feature extractors for the time-series images of vector-borne diseases. ImageNet is a well-known convolutional neural network trained on a large visual dataset comprising millions of images for image recognition tasks (Lukas et al., 2022). In our AI pipelines, ImageNet acts as a feature extractor, utilizing only the pre-trained networks, while keeping the flattened and classifying layers frozen. Consequently, the feature extractor performs the learning process and provides outputs as flattened deep features, which are then used as inputs for traditional machine learning classifiers (Ebrahim et al., 2019). 
We employed six common ImageNet architectures in the current study, namely, VGG16, ResNet50, ResNet101, ResNet152, Xception, and Efficient_NetB5. These architectures differ in their neural network designs and transfer learning algorithms (Morid et al., 2020). Utilizing these pre-trained models can offer different approaches for training vector-borne time-series images, possibly reducing training time and improving performance compared to building neural networks from scratch. For more information on ImageNet and its advantages, and disadvantages, please refer to the Keras Applications on the Keras Applications website. 

# 4.2. Traditional machine learning  
This study employed five different traditional machine learning methods as classifiers based on practical experience and general knowledge. The methods included  Logistic Regression (LR), Random Forest (RF), K-Nearest Neighbor (KNN), and Support Vector Machine (SVM) with both linear and radial basic functions (rbf), used to classify various scenarios of vector-borne diseases. 

# 4.3. Fine-tuning 
The current study utilized fine-tuning as an optimization process to enhance the performance of ImageNet’s final layers as classifiers for the target domain (i.e., vector-borne diseases). During feature extraction, the flattened and connected layers of ImageNet were removed and replaced by either new fully connected layers or by traditional machine learning methods acting as classifiers (Sarkar et al., 2018).The hyperparameters for these layers were selected through random screening based on empirical knowledge. Additionally, the GridSearch function in the Python Scikit-Learn package was employed to identify the optimal parameters for traditional machine learning classifiers (Morid et al., 2020). 

# 5. Python codes 
The Python codes uploaded in this blog cover time-series image generation using different encoding methods mentioned above, and the deployment of differnt machine learning agorithms to anticipate vector-borne diseases. 
- Image generation
- Machine learning models
- Fine-tuning enhancing models
# 6. Data and images examples



                                      





