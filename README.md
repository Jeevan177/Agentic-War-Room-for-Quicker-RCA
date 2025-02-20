# 🚀 Agentic War Room for Quicker RCA

## 📌 Overview  
Agentic War Room is an intelligent **Root Cause Analysis (RCA) chatbot** 
built using Streamlit and Machine Learning techniques. It processes 
system logs and metrics to detect anomalies and provide insights 
into potential issues.

## ✨ Features  
- 📊 **Anomaly Detection**: Uses Z-score analysis and Isolation Forest for detecting abnormal system behavior.  
- 📝 **Log Correlation**: Matches system logs with anomalies to identify root causes.  
- 🤖 **AI Chatbot Assistance**: Provides intelligent responses based on system issues dataset.  
- 📄 **Summarization Model**: Uses DistilBART to summarize issue resolutions.  
- ⚡ **Pretrained Models**: Utilizes DistilGPT-2 for chatbot interactions.  

## 🛠 Tech Stack  
- **Python** (Scikit-Learn, Pandas, NumPy, SciPy)  
- **Hugging Face Models**: DistilBART, DistilGPT-2  
- **Streamlit** (for interactive UI)  
- **FuzzyWuzzy** (for text matching)  
- **PyTorch & Transformers** (for deep learning models)  

## 🚀 How It Works  
1️⃣ **User selects an issue** from the dropdown menu in the Streamlit UI.  
2️⃣ **DistilBERT analyzes the issue**, classifies it, and predicts the potential root cause.  
3️⃣ **The predicted root cause and relevant insights** are displayed interactively in the UI for troubleshooting.  


---

 
