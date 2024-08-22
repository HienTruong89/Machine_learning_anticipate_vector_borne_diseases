# AI_pinelines_anticipate_vector_borne_diseases

# 1. Introduction
Vector-borne diseases are illness transmitted to human by infected mosquitoes, ticks, and fleas. These diseases are highly concerned by public health due to their dynamic increases corresponding to climate change.
The current project investigates the potential of different AI pinelines to anticipate vector borne diseases. Over the past decades, many studies have been conducted to predict the epidemic of these deseases, but mostly using collected time-series datapoints. In the current project, we focus on anticipating vector-borne deseases based on image-based time-series representation. The rapid development of AI generations on image recognition and detection could help to predict the outbreaks. 
# 2. AI pinelines 
AI pineline refers to structured workflows utilized to develop, deploy and maintain machine learning models. It consists of various processes including data collection, data processing, data extraction , model training, model evaluation, model deployment, monitoring and maintenance, and feedback loop. Integration of AI pinelines help ensure AI solutions manageble and that models are robust, scalable and maintainable. 

The current study explores various AI pipelines aimed at anticipating stages of vector-borne diseases. The key innovations induce employing diverse encoding methods to transform scalar time-series data into image-based representations and leveraging transfer learning from feature extractors to enhance the prediction of mosquito vector-borne outbreaks. 



![image](https://github.com/user-attachments/assets/dead9dbb-9f6b-4d17-8b82-f538de266aae)

Figure 1: Encoding scalar time series data using the recurrent plot method and a fusion of three recurrent plots into either a recurrent plot image (ReIM) or channels of a red-green-blue image (RGB), and fit into different machine learning models for prediction of tipping point or stable events.  

# 3. Image-based time-series representation methodologies
Three different encoding methods, including the recurrent plot (PR), Gramian angular field (GAF), and Markov transition field (MTF), to generate time-series image-based representations from scalar time-series data were investigated. 
For n observations, the time series x(t) is written below. 
                                                                   x(t) = { x1, x2, x3, ….,xn}

The main idea was that these encoding methods transform time-series data into high-dimensional image representations, facilitating the search for temporal correlations and transitions between data points. 

# 3.1. Encoding time-series with recurrent plot method

The recurrent plot (RP) method is a transformative technique where the elements in the matrix represent the actual distances between points in the time series (Marwan et al., 2007). This study constructed a recurrence plot image (ReIM) using the pairwise standardized Euclidean distance between elements of a time series x(t), which takes an account for the correspondent variance across the dimension of the original data (SciPy, 2008-2024). The condensed distance matrix was then converted into a square matrix form, generating a recurrent plot image (ReIM). 


                                                                   ![image](https://github.com/user-attachments/assets/427a5bb4-f4b0-4de7-9d3e-37970c1d87dc)

where dij is the distance matrix at a certain i row and j column, V_i is the dimensional variance of the time-series data for i row.

                                                                   Recurrent Plot = RP [i, j] = dij
                                      





