import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="YouTube Performance Predictor",
    page_icon="🎥",
    layout="wide"
)

# API URL
API_URL = "http://api:8000"  # Use 'api' service name in Docker, 'localhost:8000' for local

def call_api(endpoint, method="GET", data=None):
    """Call API with error handling"""
    try:
        if method == "GET":
            response = requests.get(f"{API_URL}{endpoint}", timeout=10)
        else:
            response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        # Try localhost if Docker service name fails
        try:
            api_url = "http://localhost:8000"
            if method == "GET":
                response = requests.get(f"{api_url}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{api_url}{endpoint}", json=data, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except:
            return {"error": "Cannot connect to API. Make sure the API server is running."}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# Main App
st.title("🎥 YouTube Performance Predictor")
st.markdown("Predict your video's performance before publishing!")

# Sidebar for API status
st.sidebar.title("🔧 System Status")

# Check API health
health_data = call_api("/health")
if "error" not in health_data:
    st.sidebar.success("✅ API Connected")
    st.sidebar.info(f"Models: {health_data.get('models_loaded', 0)}")
    st.sidebar.info(f"Features: {health_data.get('features_available', 0)}")
    st.sidebar.info(f"Version: {health_data.get('version', 'Unknown')}")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.error(health_data.get('error', 'Unknown error'))

# Model Info
model_info = call_api("/model-info")
if "error" not in model_info:
    st.sidebar.success("🤖 Models Loaded")
    with st.sidebar.expander("Model Details"):
        st.json(model_info.get('models', {}))

# Main prediction interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Video Details")
    
    # Video input form
    with st.form("video_form"):
        title = st.text_input(
            "Video Title", 
            value="Amazing AI Tutorial 🤖",
            help="Enter your video title (1-200 characters)"
        )
        
        description = st.text_area(
            "Description", 
            value="Learn machine learning basics in this comprehensive tutorial",
            help="Video description (optional)"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            channel_id = st.text_input("Channel ID", value="UC123456789")
            duration_seconds = st.number_input("Duration (seconds)", min_value=1, max_value=43200, value=600)
        
        with col_b:
            publish_hour = st.selectbox("Publish Hour", range(24), index=14)
            publish_day = st.selectbox(
                "Day of Week", 
                options=list(range(7)),
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                index=1
            )
        
        tags = st.text_input(
            "Tags (comma separated)", 
            value="ai, tutorial, machine learning",
            help="Enter tags separated by commas"
        )
        
        category_id = st.selectbox(
            "Category",
            options=["22", "23", "24", "25", "26", "27", "28"],
            format_func=lambda x: {
                "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
                "25": "News & Politics", "26": "Howto & Style", "27": "Education", "28": "Science & Technology"
            }.get(x, f"Category {x}"),
            index=0
        )
        
        submit_button = st.form_submit_button("🎯 Predict Performance", type="primary")

with col2:
    st.header("📊 Quick Tips")
    st.info("💡 **Optimize your video:**")
    st.markdown("""
    - Use emojis in titles 😊
    - Post during prime time (6-10 PM)
    - Keep titles 30-60 characters
    - Add relevant hashtags
    - Shorts (<60s) for viral potential
    """)

# Prediction results
if submit_button:
    if not title.strip():
        st.error("Please enter a video title")
    else:
        # Prepare data
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        video_data = {
            "title": title,
            "description": description,
            "channel_id": channel_id,
            "duration_seconds": duration_seconds,
            "publish_hour": publish_hour,
            "publish_day_of_week": publish_day,
            "tags": tag_list,
            "category_id": category_id
        }
        
        # Make prediction
        with st.spinner("🔮 Predicting performance..."):
            prediction = call_api("/predict", method="POST", data=video_data)
        
        if "error" not in prediction:
            st.success("✅ Prediction completed!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                views = prediction.get('predicted_views', 0)
                st.metric("👀 Predicted Views", f"{views:,.0f}")
            
            with col2:
                likes = prediction.get('predicted_likes', 0)
                st.metric("👍 Predicted Likes", f"{likes:,.0f}")
            
            with col3:
                comments = prediction.get('predicted_comments', 0)
                st.metric("💬 Predicted Comments", f"{comments:,.0f}")
            
            # Confidence and recommendations
            confidence = prediction.get('confidence_score', 0) * 100
            st.metric("🎯 Confidence", f"{confidence:.1f}%")
            
            # Recommendations
            recommendations = prediction.get('recommendations', [])
            if recommendations:
                st.subheader("💡 Optimization Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            
            # Performance visualization
            st.subheader("📈 Performance Breakdown")
            
            # Create a simple bar chart
            metrics_data = {
                'Metric': ['Views', 'Likes', 'Comments'],
                'Predicted': [views, likes, comments],
                'Log Scale': [np.log10(max(views, 1)), np.log10(max(likes, 1)), np.log10(max(comments, 1))]
            }
            
            import numpy as np
            
            df_metrics = pd.DataFrame(metrics_data)
            
            fig = px.bar(
                df_metrics, 
                x='Metric', 
                y='Predicted',
                title="Predicted Performance",
                color='Metric'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info
            with st.expander("🔍 Prediction Details"):
                st.json({
                    "prediction_id": prediction.get('prediction_id'),
                    "model_version": prediction.get('model_version'),
                    "confidence_score": prediction.get('confidence_score')
                })
        
        else:
            st.error(f"❌ Prediction failed: {prediction.get('error')}")

# Footer
st.markdown("---")
st.markdown("### 🚀 About This Tool")
st.markdown("""
This tool uses machine learning models trained on YouTube data to predict video performance.
The predictions are estimates based on title, timing, and content features.

**Note:** Actual performance depends on many factors including content quality, 
audience engagement, and platform algorithms.
""")

# Performance metrics (if available)
if st.checkbox("Show Model Performance"):
    if "error" not in model_info:
        st.subheader("🤖 Model Performance")
        st.json(model_info)
    else:
        st.error("Model information not available")