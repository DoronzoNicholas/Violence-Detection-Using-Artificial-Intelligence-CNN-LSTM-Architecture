
# Violence-Detection-Using-Artificial-Intelligence (CNN-LSTM)
<div align="justify">
The development of this project is based on the possibility of addressing challenges associated with the detection of violence within public space. Embracing artificial intelligence in CCTV monitoring can transform public safety and society's approach to security.

This project is trying to address this issue by investigation the potential use of artificial intelligence within video surveillance systems so that incident could potentially be responded during early stages rather than its aftermath. Monitoring CCTV footage for an operator can be a demanding task especially when the count of cameras is substantial, therefore the proactive monitoring can be in some instances impractical with the possibility to overlook critical events.

In this project i will combine 2 networks, the CNN, and the LSTM network. I will be discussing how these networks work and the benefits of using this combination.</div>

# Design and Architecture
<div align="justify">
I implemented the use of CNN architecture with the combination of LSTM (long short-term memory). Since I will be training my model on a video dataset the benefit of capturing the spatial and temporal features during the training makes this architecture an excellent fit for the purpose.
LSTM stands for long short-term memory, and it derives from the RNN (recurrent neural network). This method can produce feedback connections which makes possible to process sequence of data. The sequence of data will be represented as vectors and the sequence will be processed one at the time.
The information is passed to cells which are also known as “cell state and hidden cell”. The information’s passing through the cell can be added or removed thanks to the presence of gates, which are activated by the sigmoid function. Therefore, we can achieve the flow of the information in and out from these cells.
What type of cells we have inside the LSTM model?
The forget gate: responsible in determine which information from the previous cell state is to retain and which one is to discard. To determine which information will be retained and which one is discarded, it takes the input from the current input and the previous hidden state, the output produced will be in the range of 0 and 1 which will then passed to the sigmoid function and determine which information will be kept and discarded.
The input gate: responsible in determine what information will be added to the cell state. Takes the input from the previous hidden state and the current input and it produce a value between 0 and 1. This value will determine the portion of information which will be added to the cell state. The output of the input gate will be computed with a new candidate state using the tanh function. This update will be added to the cell state.
The output gate: responsible in determine what information should be used to compute the hidden state for the current time step. Same as the input gate values the process of the tanh function will apply to the output gate, however in this instance this will produce a new hidden state instead.
When we talk about the candidate state during input gate phase, we refer to a potential piece of information which can be added to the current cell state which for this instance will be used as a candidate. During the output gate, when the LSTM model is using the information to make a possible prediction, the candidate state will be considerate and used as a sort of hint.
The use of gates, memory cell, and the combination of sigmoid and tanh activation functions are what makes the LTSM model more suitable compared to the RNNs. Thanks to these factors we can capture long-range dependencies from the sequence of data allowing us to mitigate the vanishing gradient problem.
This problem is in fact found during the training of the model particularly during the use of a recurrent neural network (RNNs). This happens during the so-called backpropagation which consists of adjusting the weight and biases parameters of the network. If this window is becoming very small this propagates backward in the network layers the learning will become difficult.
To put this into prospective let’s imagine we would like to teach someone a task and time to time we give feedback (backpropagation) on how to improve this task, if the feedback is very vague or inconsistent (the parameters becoming small) the learning will become inefficient.
The LSTM network is advantageous while working with the sequential data tasks, such as natural language processing, time series forecasting or speech recognition, in few words is good when we must capture temporal data. However, LTSM alone is not the primary choice in computer vision task.
When we talk about computer vision we often rely on the convolutional neural network (CNNs) or known architectures such as residual networks (ResNets), MobileNet etc. These types of architectures are in fact specifically designed to manage data like images or video frames, where the spatial relationship is fundamental.
In computer vision the temporal data which can be captured by the LSTM network can be useful for different aspects of the task. In video analysis for example the temporal features are essential for the detection of motion, recognition of actions, tracking the movement of an object. When we want to detect an action, these features can give us the progression over time of the action which will allow us to determine the relationship to a specific action. On the other hand, the spatial features extracted from CNNs or related architecture, will allow us to understand the image, how it is arranged and what it looks like. For instance, if we want to extract spatial features of a tree in a picture, we can provide the CNN model with data containing a tree and the spatial features will hold the information on where the tree is on the picture and how it looks like.
Looking at these 2 characteristics, from the temporal and spatial features prospective, we can potentially fuse these 2 types of networks architecture so that we obtain an improved version of the 2.
For instance, let’s take in consideration multiple frames from a video, a sequence of frame where a person is preforming an action will be captured by the temporal features and on the other hand from each individual frame the spatial features will be extracted giving us a clear understand of the content of different part of the scene in the frame such as the road, the sky, a person etc.
# Dataset and Data Loading
Before training the model, we must find a suitable dataset. The dataset must contain relevant videos representing violent and non-violent situations. I discovered 2 suitable datasets which I combined. The decision to fuse these 2 datasets together was mainly for the type of video context each of these 2 datasets were representing. The first dataset is called” RWF2000-Video-Database-for-Violence-Detection” and contained videos representing mostly CCTV footage for violence and non-violence situations. </div>

Dataset link:
GitHub - [mchengny/RWF2000-Video-Database-for-Violence-Detection: A large scale video database for violence detection, which has 2,000 video clips containing violent or non-violent behaviours.](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)


The second dataset is called “Real Life Violence Situations Dataset” and contains videos that are more handheld footage of violence and non-violence situations. 

Dataset link:
[Real Life Violence Situations Dataset (kaggle.com)](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
<div align="justify">
I added these 2 datasets together so that I could potentially obtain more generalization from the footages contexts as well as providing more samples on what violence consists of. The combination of these 2 datasets is balanced, with the same amount of video count and average video length from both categories. 
The first option was to extract all the frames from the video and then use a function given by TensorFlow called “image_dataset_from_directory” to collect all the frames and then train my model. My goal, however, was to find a way to directly feed the video. I came across an article written by Patrice Ferlet who shared his keras-video-generator. Patrice was sharing 3 different types of functions called “VideoFrameGenerator”, “SlidingFrameGenerator” and “OpticalFlowgenerator”. 
I decided to follow and refer to the implementation of these functions and use these for my data loading process and evaluate whether this option was suitable for my task.</div>

**Article:** 
[Training a neural network with an image sequence — example with a video as input | by Patrice Ferlet | Smile Innovation | Medium](https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f)
I trained the model using the ” VideoFrameGenerator”, “SlidingFrameGenerator” approach.
![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/9d749bc9-7a77-4805-949d-a74a68575f9f)

# Model Architecture
The model will have an initial input shape of [8, 5, 224, 224, 3], where the 8 stands for the batches number, the 5 are the number of frames, 224,224 referred to height and width and finally the 3 is referred to the number of channels, in this case RGB (Red, Green, Blue).
 ![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/e2e854ad-151e-4c85-a892-6427901b729f)
<div align="justify">
The CNN architecture used is “MobileNet”, a powerful technique called transfer learning that leverages a pre-trained model which has already learned features and weights. 
The first layer is a “timedistributed” layer where we have the CNN wrapped inside. I wanted to implement the “timedistributed” layer so that we apply the CNN feature extractions on each of the frames independently. After the pictures are passed through the CCN we will obtain a shape that will consists of 8 batches of 15 frames along with a value consisting in the extracted features.
When we reach the LSTM layer this is what we will have according to my studies around this subject.
The LSTM will process the sequence of frames one by one; this is the part where the LSTM handles the temporal sequence. The LSTM will maintain the hidden state for the sequence; this is fundamental as this will allow us to collect information from past frames so that we keep tracks of dependency amongst frames. During the process we will encounter the gating mechanism which we have talked about in previous paragraphs; this will control the flow of the information and decide what information we must retain and what information to forget. During the training as the frames are processed sequentially the features obtained are abstracted and summarized; This will allow us to obtain valuable patterns which will be important for our task.
The LSTM output shape will be formed by the number of batches along with the number of LSTM units. In the architecture I decided to use 256 LSTM units, the choice of this value will determine the performance of our LSTM in capturing temporal dependency. In the later stage of this project, I twitched some values and layers in the architecture.
Before transferring to the last layers of the model we will convert this 2D tensor into 1D tensor. This is done with the “Flatten” layer. We will use the “Flatten” layer so that we can further process our data into the “Dense” layers. 
The “Dense” layers are fully connected layers and are responsible to further processing the data until the last layer where we will have just the 2 number of neurons referring to the number of classes we have for our task; This layer is where our prediction will be given. In this instance we have a binary classification for the detection of violence or non-violence category. 
Before the training part, I used some optimizers and parameters and then fit the model with my training and validation dataset.</div>
![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/1105fdd6-1baa-49fa-99c2-e45bc1db14d4)

![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/134948a6-8332-4315-ab07-827d67abeb8f)

![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/bb92dee9-87fd-4802-b11f-f154bf272c42)

# Real-time detection design
<div align="justify">
The final python code used during the test was design by following my own approach to the topic of error handling and enhancement the model reliability. The design used sees the implementation of label count rather than a threshold which is in some instances is the first choice. In machine learning what we are trying to do is to mimic the human intelligent. When we train a model, we are trying to build the relevant knowledge for the machine to perform a certain task. We cannot assume that a machine would be always correct in its response. 
Even humans with a far develop brain capability can get a wrong judgment. However, the judgment in decision making becomes easier when we weigh the information received. The goal of label count is to in fact mimic that judgment response by collecting enough information for predicting an episode. I took in consideration a first-hand example. It might happen that if we are out and we see an episode happening in front of us or in the distance, we might not have a clear prediction of what is happening. In that instance our brain is in fact collecting labels (violent/non-violent) and its count has a threshold. 
Once we reach a certain count we react to the situation accordingly. Our response might not be straight forward and in certain occasion we also need the right threshold to achieve a prediction. This is the reason why a opt for a label count basis. During the real time I collect the relevant data and set a threshold on the label count. That count will store all the violent labels, even the false positive labels. 
This list of labels will be held only for 1 minute, if in that minute we don’t have enough evidence that a violent situation is occurring then this count will be reset. During the minute some false positive will be collected however this will be handled. As we will see in result and analysis section during the occurrence of a violent episode the label count will increase considerably compared to the false positive. Therefore, if a violent episode is happening the threshold will be reached easily, and we can label that episode as violent. 
The use of label does not make the model more intelligent in identifying a violent situation, it  operates as error handling as we cannot label the whole situation as violent if we occur to have a false positive during the real time detection. </div>

# Model Testing	
## Developing a confusion Matrix for video sequence
The application of the confusion matrix is different when dealing with videos as we don’t have only one label to predict but multiples. I tested the model using a different dataset that has not related videos to the dataset used during the training.

**Testing Dataset:**
[Video Fights Dataset (kaggle.com)](https://www.kaggle.com/datasets/shreyj1729/cctv-fights-dataset/)
<div align="justify">
My testing dataset contains videos of CCTV footage and non-CCTV footage. By reviewing some video sample, I denoted that the length of the videos is not consistent and violence actions are not occurring right away, with some instances of occurrence in later stage. Therefore, many videos will contain mostly neutral labels with instances of violence.  
The approach used in the updated confusion matrix will consist in the same approach I will be using during the test of a real-time detection. I set up a threshold by keeping the count of violent label detected in the video. Therefore, I begun to work on a base line, where, when a certain count is reach, we can define that a violence situation is occurring. The problem was to find a threshold which takes count of some minor false positive which might occur during a neutral action. In fact, as denoted in previous testing there might be on some occasion where a label “violent” is produced for a short time. </div>
 
 ![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/507afa7f-6b28-4de8-9659-b06b2c383dc7)

 ![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/33f558c1-f525-4866-8a4a-03b21c4d8af9)

![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/5172a33f-6fec-4193-9a0f-34db7a03bdb4)

I undertake a confusion matrix for both the model trained with the “VideoFrameGenerator” method and the model trained with the SlidingFrameGenerator”.
**VideoFrame Model:**
![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/c445299a-9f49-4d25-bdc5-ee700a0580be)

![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/1c158f43-8764-4e26-befd-dcd8f01331f1)


I afterwards tested the second model, trained with the slidingFrame method. I produced the confusion matrix and the classification report
--SlidingFrame Model:**
![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/2159d96a-22d7-4d78-ac06-04b17095d8b0)

![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/e472f02a-5bd0-4940-bdc8-ce049923a244)
 <div align="justify">
We can denote that there were more occurrences where the model would predict a non-violent situation as violent. After viewing some video files, I noticed that slidingFrame is more sensible in producing violent label compared to the VideoFrame. Therefore, this difference gave the worst result during a non-violence situation, where a more saddle episode would produce a violent label, and better results in triggering a violent label.
We can say that the overall performance of the 2 models is not far apart and that I twitch on each specific model with a more suitable label count would compensate the additional sensitivity of the “slidingframe”. 
After we confirmed the effectiveness of the models, we can now test them on real time detection. To test the live view detection, I placed my oak-1 camera facing my TV and from there I played CCTV footage related to violent and non-violent episodes. 
For the non-violent episodes I opt for a live view from one of the cameras placed in London which can be found online. </div>

**Live video footage link:**
[EarthCam - Abbey Road Cam](https://www.earthcam.com/world/england/london/abbeyroad/?cam=abbeyroad_uk)
<div align="justify">
For the violent CCTV footage, I used the dataset used for the testing mentioned before.
The testing will be performed in the following order. I will play the live view camera for some time and then switch directly to a violent scene so that I can determine the responsiveness and the overall performance. 
On the testing code I added a threshold of 5 labels over a 1-minute time span, if the label count is more than 5 then an alarm will be triggered. The label count will be refreshed if a trigger is reached or 1 minute is passed.  </div>

**Video Frame Model:**
 ![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/000896d3-e759-40f0-9c9f-fd07f15a7d6a)
 ![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/c87279d2-6cf8-4ca9-8cc0-6b816a270100)
 
**Sliding Frame Model:**
![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/fb0a5f14-3d9c-4c98-b3e8-5f089370c3f3)

![image](https://github.com/DoronzoNicholas/Violence-detection-using-Artificial-Intelligence/assets/123806307/813cb219-b68c-4b70-8f3c-340107c110b3)

# Limitation and Further Development
<div align="justify">
In the section “results and analysis” we highlighted some limitation of the model. On some occasion a non-direct violence would not result in a violent classification by the model. The dataset used for training the model in fact represent mainly episode of direct violent situation and there is not enough data regarding an anti-social behaviour. The scope of this project is for now to address the global representation of the violence, which is found everywhere in the same form. In fact, different form of violence such as the anti-social behaviour and different action which on our eyes are categorized as violent might not be seen in the same way in other part of the world. Further limitation is with the use of weapons. The detection of a violent situation would be triggered when this would have a direct contact with a second person. Therefore, someone standing with a bat, a gun or a knife would possibly not trigger a violent classification. The are potential possibility which in turn can fill the gaps of the current model limitation. Additional training of the model addressing the first limitation addressed would in fact cover the anti-social limitation giving the model a broader understanding of the word violence in the contest of anti-social behaviour. The further limitation which consists mainly a subject standing but not directly showing violence might see the use of additional model running along with the main one with the scope of detecting object. For this instance, all potentially dangerous object detected would con trigger a violent classification.  It needs to be noted that the purpose of this project and its implementation are all related to the public oriented environment, meaning that a person holding a dangerous object should not be in a public space. Moreover, another development would be the use of audio. Since cameras are limited to only their field of view, by adding the surroundings of audio would help the detection of possible situation even in the surroundings areas. This area however might have limitation of privacy when used in public, however it might only see easier implementation in country where the privacy in this front applied differently.</div>

