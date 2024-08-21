# PolypDetection
## Introduction to the Problem
### What is a Polyp?
A polyp is an abnormal growth of tissue projecting from a mucous membrane. In the context of gastrointestinal health, polyps are commonly found in the colon and rectum. While most polyps are benign, some can develop into cancer over time, making their detection and removal crucial.

### What is a Colonoscopy?
A colonoscopy is a medical procedure used to examine the interior of the colon and rectum. During the procedure, a long, flexible tube with a camera (colonoscope) is inserted through the rectum to allow the doctor to view the entire colon. This procedure is essential for detecting polyps, cancers, and other abnormalities.

### Statistics of Missed Polyps
Colonoscopy is great when it comes to colorectal cancer screening, yet it is not perfect. Studies have shown that **22% to 28%** of polyps and **20% to 24%** of adenomas are missed during colonoscopies. This high miss rate can lead to the development of interval cancers, which are cancers that occur between regular screenings. The integration of AI in colonoscopy procedures has shown promise in reducing these miss rates. For instance, AI-assisted colonoscopies have demonstrated a significant reduction in the miss rate of precancerous polyps, from 32.4% to 15.5%.

### How AI Can Solve the Problem
Artificial Intelligence (AI) can enhance the detection of polyps during colonoscopies by using deep learning algorithms to analyze real-time video feeds. These AI systems can identify polyps that might be missed by the human eye, especially small or flat ones. By marking potential polyps in real-time, AI assists endoscopists in making more accurate diagnoses and decisions. This not only improves detection rates but also reduces the overall cost and time of the procedure.

## Other Solutions Out There
### Existing AI Tools for Polyp Detection
There are several commercial products available that utilize AI for polyp detection. One notable example is the GI Genius™ Intelligent Endoscopy Module by Medtronic. This system uses AI to assist endoscopists by highlighting potential polyps in real-time during colonoscopies. Studies have shown that the GI Genius™ module can increase adenoma detection rates by up to 14.4%. Another example is the EndoBRAIN® by Cybernet Systems, which also uses AI to improve the detection and characterization of polyps during endoscopic procedures.

## Model
### Technologies Used
To develop the polyp detection tool, I utilized several advanced machine learning technologies:

- **PyTorch:** For building and training the neural network models.
- **ViT (Vision Transformer):** A transformer-based model for image recognition tasks.
- **DINO (Self-Distillation with No Labels):** A self-supervised learning method that enhances the performance of the Vision Transformer.

### Model Architecture and Explanation
The model architecture is designed to leverage the strengths of both ViT and DINO. The spatial encoder used in this model has pretrained weights from the team I did my internship with, which were frozen during training to retain their learned features.
![image](https://github.com/user-attachments/assets/40dc672f-cfd1-44a7-a27a-2774a067e18f)


- **Encoder:** The encoder is a Vision Transformer (ViT) model with pretrained weights from the GI team. ViT models are known for their ability to capture long-range dependencies in images by treating image patches as sequences, similar to how words are treated in natural language processing. These pretrained weights are frozen to preserve the learned features and prevent overfitting during further training.
  
- **Dropout Layers:** Dropout is used to prevent overfitting by randomly setting a fraction of input units to zero during training.
  
- **Classifier:** A linear layer that maps the embeddings to the output classes (polyp or no polyp).
  
- **Loss Function:** CrossEntropyLoss is used with class weights to handle class imbalance. The class weights were calculated based on the frequency of each class in the dataset. For example, if the dataset has significantly more images without polyps than with polyps, the weight for the polyp class is increased to ensure the model pays more attention to detecting polyps. In this case, the weights were set as [0.5788, 3.6725], reflecting the imbalance in the dataset.

## Data
### Dataset: GastroVision
For this project, I used the **GastroVision** dataset, which contains a comprehensive collection of colonoscopy images. GastroVision, with two broad categories (upper GI and lower GI), covers 36 classes belonging to anatomical landmarks or pathological findings. Proper categorization of these classes can be visualized from the diagram given below.

![gastrovision5](https://github.com/user-attachments/assets/8ebf0a84-bae3-40b9-89fe-33e8e982fa44)

### Data Splitting
To prepare the data for training and testing, I created two folders: one containing images with polyps and the other containing regular colonoscopy images. The data was then split into training and testing sets using the following code:

`train_data, test_data = train_test_split(custom_dataset.data, test_size=0.2, random_state=42)
`

- Training Data: 734 images
- Testing Data: 147 images
This split corresponds to approximately 83.3% of the data for training and 16.7% for testing.

### Licence 
CC-By Attribution 4.0 International

## Training
### Weights & Biases (WandB)
To monitor the training process and track various metrics, I used **Weights & Biases (WandB)**. This tool provides a comprehensive dashboard for visualizing training progress, comparing runs, and sharing results. By integrating WandB, I was able to keep a close eye on key metrics such as loss, accuracy, precision, recall, and F1 scores.

### Monitoring Metrics
Metrics were monitored using WandB, which allowed for real-time tracking and visualization. This helped in identifying any issues early on and making necessary adjustments to the training process.

### Batch Size
The batch size used for training was **128**. This size was chosen to balance between computational efficiency and model performance.

### HuggingFace Trainer
I used the HuggingFace Trainer for training the model. The Trainer class simplifies the training loop and provides built-in support for various features such as learning rate scheduling, mixed precision training, and early stopping.

### Learning Rate Scheduler
A learning rate scheduler was used to adjust the learning rate during training. Specifically, I used a **cosine learning rate scheduler**. This type of scheduler gradually decreases the learning rate following a cosine curve, which helps in fine-tuning the model and avoiding overshooting the optimal point.

### Training Metrics
Here are the training metrics and metrics used:

- **Batch Size:** 128
- **Learning Rate:** 0.0002
- **Weight Decay:** 0.002
- **Number of Epochs:** 10
- **Evaluation Strategy:** Epoch
- **Logging Steps:** 10
- **Mixed Precision Training:** Enabled (fp16)
- **Learning Rate Scheduler:** Cosine
- **Warmup Steps:** 500
- **Gradient Accumulation Steps:** 2
- **Early Stopping:** Enabled (patience of 3 epochs)

By using these settings and tools, I aimed to ensure a robust and efficient training process, leading to a well-performing model for polyp detection.

## Challenges and Solutions
### Overfitting
One of the primary challenges I faced during the development of the model was overfitting. Overfitting occurs when the model learns the training data too well, including its noise and outliers, which negatively impacts its performance on new, unseen data.

### Solutions Implemented
To address overfitting, I implemented several techniques:
- **Weighted Loss:** Given the class imbalance in the dataset, I used a **weighted loss function**. By assigning higher weights to the minority class (polyp images), the model was encouraged to pay more attention to detecting polyps, which helped in balancing the learning process.
- **Dropout:** I added dropout layers with rates of **0.5 and 0.7** to the model to randomly set a fraction of input units to zero during training. This helps prevent the model from becoming too reliant on specific neurons, thereby improving its generalization ability.
- **Freezing the Weights:** The pretrained weights of the encoder were frozen to retain their learned features and prevent overfitting.
- **Increase the Batch Size:** Increasing the batch size helped in stabilizing the training process and improving the model’s performance.
- **Increase the Learning Rate:** Adjusting the learning rate helped in speeding up the convergence of the model.
- **Added Data to Training:** Adding more data to the training set improved the model’s ability to generalize to new, unseen data.
- **Learning Rate Scheduler:** Initially, a **linear** learning rate scheduler was used, which was later changed to a **cosine** scheduler to better adjust the learning rate during training.
- **Increased Weight Decay:** The weight decay was increased to 0.002 to regularize the model and prevent overfitting.
- **Early Stopping Callback:** An early stopping callback was added to stop training when the model’s performance stopped improving, preventing overfitting.
- **Gradient Accumulation:** Gradient accumulation was used to effectively increase the batch size without requiring more memory, which helped in stabilizing the training process.

By implementing these strategies, I was able to mitigate overfitting and improve the model’s performance on the test data.

## Results
### Confusion maMtrix
The confusion matrix provides a detailed breakdown of the model’s performance on the test set at epoch 5:

<img width="143" alt="image" src="https://github.com/user-attachments/assets/5842292c-d317-4583-afba-66a5841bcb79">

### Evaluation Loss
The evaluation loss, as tracked by WandB, shows a consistent decrease over the training epochs, indicating that the model is learning effectively and improving its performance. Below is the graph of the evaluation loss:

<img width="284" alt="image" src="https://github.com/user-attachments/assets/4785f8d8-c0ec-4aff-a81e-46e2eae604d8">

### Classification Report
The classification report provides a comprehensive overview of the model’s performance across various metrics. Here are the results from the final epoch of training:

- Epoch: 9
- Training Loss: 0.434900
- Validation Loss: 0.308287
- Accuracy: 0.84
  
**F1-Score:**
- Class 0 (Normal): 0.80
- Class 1 (Polyp): 0.86
