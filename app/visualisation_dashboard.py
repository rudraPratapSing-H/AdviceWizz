"""
Streamlit Dashboard for RAG vs Non-RAG Comparison Visualization

Run with: streamlit run visualize_comparison.py
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="RAG Comparison Dashboard",
    page_icon="📊",
    layout="wide"
)

JSON_PATH = r"c:\Projects\AdviceWizz\AdviceWizz\app\rag_comparison_results.json"
# Load data
@st.cache_data
def load_comparison_data():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ rag_comparison_results.json not found. Please run compare_rag.py first!")
        return []

data = load_comparison_data()

if not data:
    st.stop()

# Title
st.title("🔍 RAG vs Non-RAG Comparison Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")
query_filter = st.sidebar.selectbox(
    "Select Query",
    options=["All"] + [item["query"] for item in data],
    index=0
)

# Filter data
filtered_data = data if query_filter == "All" else [item for item in data if item["query"] == query_filter]

# Metrics Section
st.header("📈 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

# Calculate averages
avg_llama3_time = sum(item["non_rag_llama3"]["time_taken"] for item in filtered_data) / len(filtered_data)
avg_gemini_time = sum(item["gemini"]["time_taken"] for item in filtered_data if "error" not in item["gemini"]) / len([item for item in filtered_data if "error" not in item["gemini"]])
avg_rag_time = sum(item["rag"]["time_taken"] for item in filtered_data) / len(filtered_data)

col1.metric("Total Queries", len(filtered_data))
col2.metric("Avg Llama3 Time", f"{avg_llama3_time:.2f}s")
col3.metric("Avg Gemini Time", f"{avg_gemini_time:.2f}s")
col4.metric("Avg RAG Time", f"{avg_rag_time:.2f}s")

# Response Time Comparison Chart
st.header("⏱️ Response Time Comparison")

time_data = []
for item in filtered_data:
    time_data.append({
        "Query": item["query"][:50] + "...",
        "Llama3 (No RAG)": item["non_rag_llama3"]["time_taken"],
        "Gemini API": item["gemini"]["time_taken"],
        "RAG (Semantic)": item["rag"]["time_taken"]
    })

df_time = pd.DataFrame(time_data)
df_time_melted = df_time.melt(id_vars=["Query"], var_name="Model", value_name="Time (seconds)")

fig_time = px.bar(
    df_time_melted,
    x="Query",
    y="Time (seconds)",
    color="Model",
    barmode="group",
    title="Response Time by Model",
    color_discrete_map={
        "Llama3 (No RAG)": "#FF6B6B",
        "Gemini API": "#4ECDC4",
        "RAG (Semantic)": "#95E1D3"
    }
)
fig_time.update_layout(xaxis_tickangle=-45, height=500)
st.plotly_chart(fig_time, use_container_width=True)

# Response Length Comparison
st.header("📏 Response Length Comparison")

length_data = []
for item in filtered_data:
    length_data.append({
        "Query": item["query"][:50] + "...",
        "Llama3 (No RAG)": len(item["non_rag_llama3"]["response"]),
        "Gemini API": len(item["gemini"]["response"]),
        "RAG (Semantic)": len(item["rag"]["response"])
    })

df_length = pd.DataFrame(length_data)
df_length_melted = df_length.melt(id_vars=["Query"], var_name="Model", value_name="Response Length (chars)")

fig_length = px.bar(
    df_length_melted,
    x="Query",
    y="Response Length (chars)",
    color="Model",
    barmode="group",
    title="Response Length by Model",
    color_discrete_map={
        "Llama3 (No RAG)": "#FF6B6B",
        "Gemini API": "#4ECDC4",
        "RAG (Semantic)": "#95E1D3"
    }
)
fig_length.update_layout(xaxis_tickangle=-45, height=500)
st.plotly_chart(fig_length, use_container_width=True)

# Detailed Response Comparison
st.header("🔎 Detailed Response Comparison")

selected_query = st.selectbox(
    "Select a query to compare responses:",
    options=[item["query"] for item in filtered_data]
)

selected_item = next(item for item in filtered_data if item["query"] == selected_query)

st.subheader(f"Query: {selected_query}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔴 Llama3 (No RAG)")
    st.markdown(f"**Time:** {selected_item['non_rag_llama3']['time_taken']:.2f}s")
    st.markdown(f"**Length:** {len(selected_item['non_rag_llama3']['response'])} chars")
    with st.expander("View Response"):
        st.write(selected_item["non_rag_llama3"]["response"])

with col2:
    st.markdown("### 🔵 Gemini API")
    st.markdown(f"**Time:** {selected_item['gemini']['time_taken']:.2f}s")
    st.markdown(f"**Length:** {len(selected_item['gemini']['response'])} chars")
    with st.expander("View Response"):
        st.write(selected_item["gemini"]["response"])

with col3:
    st.markdown("### 🟢 RAG (Semantic)")
    st.markdown(f"**Time:** {selected_item['rag']['time_taken']:.2f}s")
    st.markdown(f"**Length:** {len(selected_item['rag']['response'])} chars")
    with st.expander("View Response"):
        st.write(selected_item["rag"]["response"])
    
    if "retrieved_sources" in selected_item["rag"]:
        st.markdown("**📚 Retrieved Sources:**")
        for i, source in enumerate(selected_item["rag"]["retrieved_sources"], 1):
            st.markdown(f"**{i}. {source['book']}** (Page {source['page']})")
            with st.expander(f"View Chunk {i}"):
                st.text(source["chunk"][:300] + "...")

# Source Book Distribution (RAG only)
st.header("📚 Source Book Distribution (RAG)")

source_books = []
for item in filtered_data:
    if "retrieved_sources" in item["rag"]:
        for source in item["rag"]["retrieved_sources"]:
            source_books.append(source["book"])

if source_books:
    book_counts = pd.Series(source_books).value_counts()
    
    fig_books = px.pie(
        values=book_counts.values,
        names=book_counts.index,
        title="Distribution of Retrieved Books",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_books, use_container_width=True)

# Emotion Analysis (RAG only)
st.header("😊 Emotion Detection (RAG)")

emotions = []
for item in filtered_data:
    if "emotion" in item["rag"]:
        emotions.append({
            "Query": item["query"][:40] + "...",
            "Emotion": item["rag"]["emotion"][:100] + "..."
        })

if emotions:
    df_emotions = pd.DataFrame(emotions)
    st.dataframe(df_emotions, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")