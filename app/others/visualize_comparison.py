"""
Streamlit Dashboard for RAG vs Non-RAG Comparison Visualization

Run with: streamlit run visualize_comparison.py
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="RAG Comparison Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_comparison_data():
    try:
        with open("../Json/rag_comparison_results.json", "r", encoding="utf-8") as f:
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

# Calculate averages (excluding errors)
valid_llama3 = [item["non_rag_llama3"]["time_taken"] for item in filtered_data if "error" not in item["non_rag_llama3"]]
valid_gemini = [item["gemini"]["time_taken"] for item in filtered_data if "error" not in item["gemini"]]
valid_rag = [item["rag"]["time_taken"] for item in filtered_data if "error" not in item["rag"]]

avg_llama3_time = sum(valid_llama3) / len(valid_llama3) if valid_llama3 else 0
avg_gemini_time = sum(valid_gemini) / len(valid_gemini) if valid_gemini else 0
avg_rag_time = sum(valid_rag) / len(valid_rag) if valid_rag else 0

col1.metric("Total Queries", len(filtered_data))
col2.metric("Avg Llama3 Time", f"{avg_llama3_time:.2f}s")
col3.metric("Avg Gemini Time", f"{avg_gemini_time:.2f}s")
col4.metric("Avg RAG Time", f"{avg_rag_time:.2f}s")

# Response Time Comparison Chart
st.header("⏱️ Response Time Comparison")

time_data = []
for item in filtered_data:
    query_short = item["query"][:50] + "..." if len(item["query"]) > 50 else item["query"]
    
    time_data.append({
        "Query": query_short,
        "Model": "Llama3 (No RAG)",
        "Time (seconds)": item["non_rag_llama3"]["time_taken"] if "error" not in item["non_rag_llama3"] else 0
    })
    time_data.append({
        "Query": query_short,
        "Model": "Gemini API",
        "Time (seconds)": item["gemini"]["time_taken"] if "error" not in item["gemini"] else 0
    })
    time_data.append({
        "Query": query_short,
        "Model": "RAG (Semantic)",
        "Time (seconds)": item["rag"]["time_taken"] if "error" not in item["rag"] else 0
    })

df_time = pd.DataFrame(time_data)

fig_time = px.bar(
    df_time,
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

# Average Response Time Pie Chart
st.header("🥧 Average Response Time Distribution")

avg_times = {
    "Llama3 (No RAG)": avg_llama3_time,
    "Gemini API": avg_gemini_time,
    "RAG (Semantic)": avg_rag_time
}

fig_pie = px.pie(
    values=list(avg_times.values()),
    names=list(avg_times.keys()),
    title="Average Response Time Distribution",
    color_discrete_map={
        "Llama3 (No RAG)": "#FF6B6B",
        "Gemini API": "#4ECDC4",
        "RAG (Semantic)": "#95E1D3"
    }
)
st.plotly_chart(fig_pie, use_container_width=True)

# Response Length Comparison
st.header("📏 Response Length Comparison")

length_data = []
for item in filtered_data:
    query_short = item["query"][:50] + "..." if len(item["query"]) > 50 else item["query"]
    
    length_data.append({
        "Query": query_short,
        "Model": "Llama3 (No RAG)",
        "Response Length (chars)": len(item["non_rag_llama3"]["response"]) if "error" not in item["non_rag_llama3"] else 0
    })
    length_data.append({
        "Query": query_short,
        "Model": "Gemini API",
        "Response Length (chars)": len(item["gemini"]["response"]) if "error" not in item["gemini"] else 0
    })
    length_data.append({
        "Query": query_short,
        "Model": "RAG (Semantic)",
        "Response Length (chars)": len(item["rag"]["response"]) if "error" not in item["rag"] else 0
    })

df_length = pd.DataFrame(length_data)

fig_length = px.bar(
    df_length,
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

# Speed Comparison - Line Chart
st.header("🚀 Speed Comparison Across Queries")

speed_data = []
for i, item in enumerate(filtered_data, 1):
    speed_data.append({
        "Query #": i,
        "Llama3 (No RAG)": item["non_rag_llama3"]["time_taken"] if "error" not in item["non_rag_llama3"] else 0,
        "Gemini API": item["gemini"]["time_taken"] if "error" not in item["gemini"] else 0,
        "RAG (Semantic)": item["rag"]["time_taken"] if "error" not in item["rag"] else 0
    })

df_speed = pd.DataFrame(speed_data)
df_speed_melted = df_speed.melt(id_vars=["Query #"], var_name="Model", value_name="Time (seconds)")

fig_speed = px.line(
    df_speed_melted,
    x="Query #",
    y="Time (seconds)",
    color="Model",
    markers=True,
    title="Response Speed Trend",
    color_discrete_map={
        "Llama3 (No RAG)": "#FF6B6B",
        "Gemini API": "#4ECDC4",
        "RAG (Semantic)": "#95E1D3"
    }
)
st.plotly_chart(fig_speed, use_container_width=True)

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
        if "error" in selected_item["non_rag_llama3"]:
            st.error(selected_item["non_rag_llama3"]["response"])
        else:
            st.write(selected_item["non_rag_llama3"]["response"])

with col2:
    st.markdown("### 🔵 Gemini API")
    st.markdown(f"**Time:** {selected_item['gemini']['time_taken']:.2f}s")
    st.markdown(f"**Length:** {len(selected_item['gemini']['response'])} chars")
    with st.expander("View Response"):
        if "error" in selected_item["gemini"]:
            st.error(selected_item["gemini"]["response"])
        else:
            st.write(selected_item["gemini"]["response"])

with col3:
    st.markdown("### 🟢 RAG (Semantic)")
    st.markdown(f"**Time:** {selected_item['rag']['time_taken']:.2f}s")
    st.markdown(f"**Length:** {len(selected_item['rag']['response'])} chars")
    with st.expander("View Response"):
        if "error" in selected_item["rag"]:
            st.error(selected_item["rag"]["response"])
        else:
            st.write(selected_item["rag"]["response"])
    
    if "retrieved_sources" in selected_item["rag"]:
        st.markdown("**📚 Retrieved Sources:**")
        for i, source in enumerate(selected_item["rag"]["retrieved_sources"], 1):
            st.markdown(f"**{i}. {source['book']}** (Page {source['page']})")
            with st.expander(f"View Chunk {i}"):
                st.text(source["chunk"][:300] + "..." if len(source["chunk"]) > 300 else source["chunk"])

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
else:
    st.info("No retrieved sources found in RAG responses.")

# Model Performance Summary Table
st.header("📊 Model Performance Summary")

summary_data = {
    "Model": ["Llama3 (No RAG)", "Gemini API", "RAG (Semantic)"],
    "Avg Response Time (s)": [
        f"{avg_llama3_time:.2f}",
        f"{avg_gemini_time:.2f}",
        f"{avg_rag_time:.2f}"
    ],
    "Success Rate": [
        f"{len([item for item in filtered_data if 'error' not in item['non_rag_llama3']]) / len(filtered_data) * 100:.1f}%",
        f"{len([item for item in filtered_data if 'error' not in item['gemini']]) / len(filtered_data) * 100:.1f}%",
        f"{len([item for item in filtered_data if 'error' not in item['rag']]) / len(filtered_data) * 100:.1f}%"
    ]
}

df_summary = pd.DataFrame(summary_data)
st.table(df_summary)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data updates automatically when you run compare_rag.py")
