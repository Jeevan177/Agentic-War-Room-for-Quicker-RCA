import streamlit as st
import pandas as pd
import numpy as np
import ast
from fuzzywuzzy import process
from transformers import pipeline
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

# Load Summarization Model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load Issues Dataset
@st.cache_data
def load_issues_dataset(file_path):
    return pd.read_csv(file_path)

# Load Logs Data
@st.cache_data
def preprocess_logs(log_file):
    logs = []
    # Read file from UploadedFile object
    for line in log_file.getvalue().decode("utf-8").splitlines():
        parts = line.strip().split(" ", 3)
        if len(parts) == 4:
            logs.append({
                "timestamp": parts[0],
                "service": parts[1],
                "log_level": parts[2],
                "message": parts[3]
            })
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Load Metrics Data
@st.cache_data
def preprocess_metrics(metrics_file):
    df = pd.read_csv(metrics_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Anomaly Detection
def detect_anomalies(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = np.abs(zscore(df[numeric_cols]))
    z_anomalies = (z_scores > 2.5).sum(axis=1) > 0
    model = IsolationForest(contamination=0.05, random_state=42)
    iso_anomalies = model.fit_predict(df[numeric_cols]) == -1
    df['is_anomaly'] = z_anomalies | iso_anomalies
    return df[df['is_anomaly']]

# Correlate Logs with Anomalies
def correlate_anomalies(logs, anomalies):
    correlated_issues = []
    for _, anomaly in anomalies.iterrows():
        anomaly_time = anomaly['timestamp']
        related_logs = logs[
            (logs['timestamp'] >= anomaly_time - pd.Timedelta(minutes=5)) &
            (logs['timestamp'] <= anomaly_time + pd.Timedelta(minutes=5))
        ]
        if not related_logs.empty:
            correlated_issues.append({
                "metric_anomaly": anomaly.to_dict(),
                "related_logs": related_logs.to_dict(orient='records')
            })
    return correlated_issues

# Load Pretrained Chatbot (DistilBERT-based)
chatbot = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization")

def chatbot_response(user_query, anomalies, issues_df):
    for anomaly in anomalies:
        if any(keyword in user_query.lower() for keyword in ["slow", "error", "high", "issue", "problem"]):
            anomaly_data = anomaly["metric_anomaly"]
            anomaly_time = anomaly_data.get("timestamp", "Unknown time")
            
            metric_name = next((col for col in anomaly_data.keys() if col != "timestamp"), "Unknown Metric")
            
            related_log_message = "No logs found"
            if anomaly["related_logs"]:
                related_log_message = anomaly["related_logs"][0].get("message", "Unknown cause")
            
            # Find the best matching issue from `issues_dataset.csv`
            match = process.extractOne(metric_name, issues_df['Issue'], score_cutoff=60)  # Adjust cutoff as needed

            # Handle the case when no match is found
            if match:
                # Access elements of match using indexing if it has more than 2 values
                best_match = match[0] if match else None 
                score = match[1] if match else 0

                solutions = issues_df.loc[issues_df['Issue'] == best_match, 'Solutions'].values[0]

                # Convert string representation of list to an actual list
                if isinstance(solutions, str) and solutions.startswith("[") and solutions.endswith("]"):
                    try:
                        solutions = ast.literal_eval(solutions)  # Safely convert string to list
                    except (ValueError, SyntaxError):
                        solutions = [solutions]  # Keep it as a list with one element if conversion fails

                # Format solutions as a readable string
                formatted_solution = "\n- " + "\n- ".join(solutions) if isinstance(solutions, list) else solutions

                # Summarize if too long
                if len(formatted_solution.split()) > 30:
                    summarized_solution = summarizer(formatted_solution, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
                else:
                    summarized_solution = formatted_solution

            else:
                best_match = None  # or any other default value
                score = 0  # or any other default value
                summarized_solution = "No documented solution found. Please investigate manually."
          
            return f"Detected anomaly in {metric_name} at {anomaly_time}.\nPossible cause: {related_log_message}\nSuggested solution: {summarized_solution}"
    
    return chatbot(user_query, max_length=50)[0]['generated_text']
    
# Streamlit UI
st.title("Agentic War Room - RCA Chatbot 🤖")
st.write("Ask questions about system issues, anomalies, and root causes.")

# Upload Files
logs_file = st.file_uploader("Upload Logs File", type=["txt"])
metrics_file = st.file_uploader("Upload Metrics CSV", type=["csv"])
issues_file = st.file_uploader("Upload Issues Dataset", type=["csv"])

if logs_file and metrics_file and issues_file:
    logs_df = preprocess_logs(logs_file)
    metrics_df = preprocess_metrics(metrics_file)
    issues_df = load_issues_dataset(issues_file)

    anomalies_df = detect_anomalies(metrics_df)
    correlated_issues = correlate_anomalies(logs_df, anomalies_df)

    st.success("Files processed successfully! You can now chat.")

    # Chatbot UI
    user_query = st.text_input("Ask your question:")
    if user_query:
        response = chatbot_response(user_query, correlated_issues, issues_df)
        st.write(f"**Chatbot:** {response}")

    # Display Detected Anomalies
    if not anomalies_df.empty:
        st.subheader("Detected Anomalies")
        st.dataframe(anomalies_df)

    # Display Related Logs
    if correlated_issues:
        st.subheader("Related Log Messages")
        for issue in correlated_issues:
            st.write(f"**Anomaly:** {issue['metric_anomaly']['timestamp']}")
            st.write(f"**Related Log:** {issue['related_logs'][0]['message']}")
