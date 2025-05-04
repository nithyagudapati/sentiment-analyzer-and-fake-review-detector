## **Sentiment Analyzer and Fake Review Detector**  

### **Introduction**  

This project is a Streamlit web application that performs sentiment analysis and fake review detection on product reviews using pre-trained deep learning models. Users can input review text through the app, and the system will classify the review as positive/negative and genuine/fake. The goal is to assist users in evaluating product feedback efficiently using natural language processing.

### **Dataset**  
https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews


### **Project Overview**  
This end-to-end project consists of the following steps:

1. Data Preprocessing: Text normalization, tokenization, padding and label encoding for optimal model performance. 

2. Model Building: LSTM-based deep learning models trained for:

- Sentiment analysis - Predicts sentiment of the review (positive or negative).

- Fake review detection -  Detects whether the review is genuine or fake.

3. Model Saving: Models saved as .keras and tokenizer as .pkl for deployment.

4. Streamlit Web App: User-friendly UI to enter review text and get predictions.

5. Model Deployment: Application deployed on Streamlit Cloud.

6. Real-Time Prediction: App provides instant classification upon input.

### **Live Demo**  
Try the app on Streamlit Cloud:  
https://sentiment-analyzer-and-fake-review-detector-roghawbcxrdfmxf9zk.streamlit.app/

### **Project Directory Structure**  
sentimentanalysis_app/  
├── app.py → Streamlit application  
├── requirements.txt → Python dependencies  
├── fake_review_lstm_model.keras → Model for fake review detection  
├── sentiment_lstm_best_model.keras → Model for sentiment analysis  
├── sentiment_tokenizer.pkl → Tokenizer used for preprocessing  
├── .gitignore  
└── .gitattributes  

### **Setup Instructions**  
**Step 1:** Clone the Repository  

       git clone https://github.com/nithyagudapati/sentiment-analyzer-and-fake-review-detector.git  
       cd sentiment-analyzer-and-fake-review-detector  

**Step 2:** Create a Virtual Environment  

    For Windows:  
    python -m venv ven  
    venv\Scripts\activate   

    For macOS/Linux:  
    python3 -m venv venv  
    source venv/bin/activate  

**Step 3:** Install Required Packages   

       pip install -r requirements.txt  

**Step 4:** Run the Application   

       streamlit run app.py  

Then open http://localhost:8501 in your browser.  

### **Web Application Features**    
Review Input: Users can enter any product review.   
 **Note:** Input should be given with proper words as the model trained with proper data.

### **Dual Classification:**    

Sentiment: Positive or Negative  

Authenticity: Genuine or Fake  

Instant Output: Model responds in real-time with clear labels.  

### **Model Overview**
- Sentiment Model: LSTM model trained to predict whether a review is positive or negative.
- Fake Review Model: LSTM model trained to identify fake reviews.
- Tokenizer: Pre-fitted tokenizer used to preprocess input text for both models.

All model and tokenizer files are included in the project and loaded at runtime.

### **Deployment (Streamlit Cloud)**  
1. Push this project to your GitHub repository.  

2. Go to https://streamlit.io/cloud.

3. Connect your GitHub account and select the repository.

4. Set app.py as the entry point and deploy.

5. Share the link to let others try it online.

### **Technologies Used**
- Python

- Streamlit

- TensorFlow/Keras

- scikit-learn

- pandas

### **Problem Statement**  
- **Fake Reviews:** Online platforms are flooded with misleading or fake reviews that affect consumer trust and business reputation.  
- **Sentiment Ambiguity:** It's challenging to manually interpret the expressed opinion (positive, negative, or neutral) of large volumes of user feedback.  
- **Decision Making:** Consumers and businesses need reliable tools to evaluate review authenticity and sentiment to make better decisions.  

### **Future Enhancements**    
→ Integrate pre-trained models like BERT to boost detection accuracy.

→ Add multilingual support and real-time review analysis.

→ Deploy with Docker and enhance UI with explainable predictions.
