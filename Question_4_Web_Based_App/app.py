import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import joblib
import os
from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import random
import matplotlib.colors as mcolors

# Page configuration
st.set_page_config(
    page_title="News Cluster Explorer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .article-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
    .article-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
    }
    .article-url {
        font-size: 14px;
        color: #1e88e5;
        word-break: break-all;
    }
    .cluster-header {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .main-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #ffff99;
        padding: 2px 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'df' not in st.session_state:
    st.session_state.df = None

if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
    
if 'search_term' not in st.session_state:
    st.session_state.search_term = ""

if 'current_page' not in st.session_state:
    st.session_state.current_page = 1


def clean_text(text):
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove special characters and preserve spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower().strip()
        return text
    return ''

# Load models
@st.cache_resource
def load_models():
    try:
        # Load the vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load the SVD model
        svd = joblib.load('models/svd_model.joblib')
        
        # Load the KMeans model
        kmeans = joblib.load('models/kmeans_model.joblib')
        
        # Load the cluster to category mapping
        with open('models/cluster_to_category.pkl', 'rb') as f:
            cluster_to_category = pickle.load(f)
            
        return vectorizer, svd, kmeans, cluster_to_category
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def predict_category(text, vectorizer, svd, kmeans_model, cluster_mapping):
    """Predict the category of a news article or headline based on content"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Transform to TF-IDF features
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Apply dimensionality reduction
    text_reduced = svd.transform(text_tfidf)
    
    # Predict the cluster
    cluster = kmeans_model.predict(text_reduced)[0]
    
    # Map the cluster to a category
    category = cluster_mapping.get(cluster, "Unknown")
    
    return category, cluster


def get_domain(url):
    '''get domain from URL'''
    try:
        parsed_uri = urlparse(url)
        domain = '{uri.netloc}'.format(uri=parsed_uri)
        # Remove 'www.' if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "unknown"


def visualize_clusters(df, kmeans, svd, reduced_data=None):
    '''Visualize  the  clusters'''
    if reduced_data is None and 'cleaned_content' not in df.columns:
        st.warning("No reduced data or cleaned content available for visualization.")
        return None
    
    if reduced_data is None:
        sample_size = min(2000, len(df))
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
        st.info("Computing visualization. This may take a moment...")
        # TSNE is computationally expensive, SVD-reduced data is a better option
        reduced_data = df_sample['svd_features'].tolist() if 'svd_features' in df_sample.columns else None
    
    # t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) - 1), 
                n_iter=1000, init='pca')
    tsne_results = tsne.fit_transform(np.array(reduced_data))
    
   
    vis_df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'cluster': df['cluster'].values[:len(tsne_results)],
        'category': df['category'].values[:len(tsne_results)],
        'headline': df['headline'].fillna('No headline').values[:len(tsne_results)]
    })
    

    category_colors = {
        "Business": "#1e88e5",
        "Politics": "#d81b60",
        "Arts/Culture/Celebrities": "#8e24aa",
        "Sports": "#43a047"
    }

    fig = px.scatter(
        vis_df, 
        x='x', 
        y='y', 
        color='category',
        color_discrete_map=category_colors,
        hover_data=['headline'],
        title='Article Clusters Visualization',
        labels={'x': 't-SNE dimension 1', 'y': 't-SNE dimension 2'},
        opacity=0.7
    )
    

    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        legend_title_text='Category',
        plot_bgcolor='white',
        height=600
    )
    
    return fig


@st.cache_data
def load_news_data(file_path):
    '''Load and preprocess news data'''
    try:
        df = pd.read_csv(file_path)

        required_columns = ['headline', 'article_url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in CSV: {', '.join(missing_columns)}")
            return None
        df['domain'] = df['article_url'].apply(get_domain)
        if 'cluster' not in df.columns or 'category' not in df.columns:
            st.warning("Cluster or category information missing. Classification will be performed.")
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_data_callback():
    """Callback functions for session state"""
    file_path = None
    if st.session_state.uploaded_file is not None:
        with open("temp_upload.csv", "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())
        file_path = "temp_upload.csv"
    elif os.path.exists(st.session_state.local_path):
        file_path = st.session_state.local_path
    else:
        st.error(f"File not found: {st.session_state.local_path}")
        st.session_state.data_loaded = False
        return
    
    # Loading the news data
    df = load_news_data(file_path)
    
    if df is not None:
        st.session_state.df = df
        st.session_state.data_loaded = True
        if 'cluster' in df.columns and len(df['cluster'].unique()) > 0:
            st.session_state.selected_cluster = sorted(df['cluster'].unique())[0]
    else:
        st.session_state.data_loaded = False

def select_cluster_callback():
    """Callback for cluster selection"""
    # Reset page number when changing clusters
    st.session_state.current_page = 1

def search_callback():
    """Callback for search term changes"""
    # Reset page number when search term changes
    st.session_state.current_page = 1

def page_change_callback():
    """Callback for page number changes"""
    # Ensure page number is within valid range
    if st.session_state.df is not None and st.session_state.selected_cluster is not None:
        cluster_df = st.session_state.df[st.session_state.df['cluster'] == st.session_state.selected_cluster].copy()
        
        # Filter by search term if present
        if st.session_state.search_term:
            search_results = cluster_df[
                cluster_df['headline'].str.contains(st.session_state.search_term, case=False, na=False) |
                cluster_df['article_url'].str.contains(st.session_state.search_term, case=False, na=False)
            ]
            display_df = search_results
        else:
            display_df = cluster_df
            
        # Calculate max pages
        items_per_page = 20
        total_pages = (len(display_df) - 1) // items_per_page + 1
        
        # Ensure page is within bounds
        if st.session_state.current_page < 1:
            st.session_state.current_page = 1
        elif st.session_state.current_page > total_pages:
            st.session_state.current_page = total_pages

def main():
    st.markdown("<h1 class='main-header'>📰 News Cluster Explorer</h1>", unsafe_allow_html=True)
    
    # Load models
    vectorizer, svd, kmeans, cluster_to_category = load_models()
    
    if vectorizer is None or svd is None or kmeans is None or cluster_to_category is None:
        st.error("Failed to load models. Please check that model files exist in the 'models/' directory.")
        return
    
    # Sidebar for data loading and filters
    with st.sidebar:
        st.header("Data Options")
        st.file_uploader("Upload a CSV file with news articles", type=['csv'], key="uploaded_file")
        st.text_input("Or enter path to a local CSV file:", "categorized_news.csv", key="local_path")
        
        st.button("Load Data", on_click=load_data_callback)
    

    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"Loaded {len(df)} articles")
        if 'cluster' not in df.columns or 'category' not in df.columns:
            with st.spinner("Classifying articles... This may take a while."):
                # Add cleaned content if needed for classification
                if 'article_content' in df.columns:
                    df['cleaned_content'] = df['article_content'].apply(clean_text)
                else:
                    # Use headline if content is not available
                    df['cleaned_content'] = df['headline'].apply(clean_text)
                
                # Classify each article
                clusters = []
                categories = []
                
                for idx, row in df.iterrows():
                    text = row['cleaned_content']
                    category, cluster = predict_category(text, vectorizer, svd, kmeans, cluster_to_category)
                    clusters.append(cluster)
                    categories.append(category)
                
                df['cluster'] = clusters
                df['category'] = categories
                
                # Update session state with classified data
                st.session_state.df = df
                
                # Set default cluster selection
                if len(df['cluster'].unique()) > 0 and st.session_state.selected_cluster is None:
                    st.session_state.selected_cluster = sorted(df['cluster'].unique())[0]
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Cluster Overview", "Cluster Details", "Visualization"])
        
        with tab1:
            st.header("Cluster Overview")
            
   
            cluster_stats = df.groupby(['cluster', 'category']).agg(
                article_count=('headline', 'count'),
                domains=('domain', lambda x: len(set(x)))
            ).reset_index()
            
            # Format as a nice table
            st.subheader("Cluster Statistics")
            for _, row in cluster_stats.iterrows():
                cluster = row['cluster']
                category = row['category']
                count = row['article_count']
                domain_count = row['domains']
                
                # Get category-specific color
                category_colors = {
                    "Business": "#1e88e5",
                    "Politics": "#d81b60",
                    "Arts/Culture/Celebrities": "#8e24aa",
                    "Sports": "#43a047"
                }
                color = category_colors.get(category, "#ff9800")
                
                st.markdown(
                    f"""
                    <div style="background-color:{color}20; padding:15px; border-radius:10px; margin-bottom:15px; border-left:5px solid {color}">
                        <h3>Cluster {cluster}: {category}</h3>
                        <p><b>Articles:</b> {count} | <b>Unique Sources:</b> {domain_count}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Show distribution chart
            st.subheader("Category Distribution")
            fig = px.pie(
                df, 
                names='category', 
                color='category',
                color_discrete_map={
                    "Business": "#1e88e5",
                    "Politics": "#d81b60",
                    "Arts/Culture/Celebrities": "#8e24aa",
                    "Sports": "#43a047"
                },
                title='Distribution of Articles by Category'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Cluster Details")
            
            # Create cluster selector
            cluster_options = sorted(df['cluster'].unique())
            
            # Use selectbox with on_change callback
            st.selectbox(
                "Select a cluster to explore:", 
                options=cluster_options, 
                key="selected_cluster",
                on_change=select_cluster_callback
            )
            
   
            if st.session_state.selected_cluster is not None:
                cluster_df = df[df['cluster'] == st.session_state.selected_cluster].copy()
                
                if len(cluster_df) > 0:
                    category = cluster_df['category'].iloc[0]
                    st.markdown(f"### Cluster {st.session_state.selected_cluster}: {category}")
                    st.markdown(f"**Number of articles:** {len(cluster_df)}")
                    
                    domain_counts = cluster_df['domain'].value_counts().head(10)
                    st.subheader("Top Sources")
                    source_fig = px.bar(
                        x=domain_counts.index, 
                        y=domain_counts.values,
                        labels={'x': 'Source', 'y': 'Number of Articles'},
                        color_discrete_sequence=[category_colors.get(category, "#ff9800")]
                    )
                    st.plotly_chart(source_fig, use_container_width=True)
                    st.subheader("Articles in this Cluster")
                    st.text_input(
                        "Search within this cluster:", 
                        value=st.session_state.search_term,
                        key="search_term",
                        on_change=search_callback
                    )
                    
                    # Filter by search term
                    if st.session_state.search_term:
                        search_results = cluster_df[
                            cluster_df['headline'].str.contains(st.session_state.search_term, case=False, na=False) |
                            cluster_df['article_url'].str.contains(st.session_state.search_term, case=False, na=False)
                        ]
                        display_df = search_results
                        st.write(f"Found {len(display_df)} results for '{st.session_state.search_term}'")
                    else:
                        display_df = cluster_df

                    with st.container():
                        display_df = display_df.sort_values('domain')
                        items_per_page = 20
                        total_pages = (len(display_df) - 1) // items_per_page + 1
                        
                        if total_pages > 1:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                if st.session_state.current_page > 1:
                                    if st.button("← Previous Page"):
                                        st.session_state.current_page -= 1
                                        st.experimental_rerun()
                            
                            with col2:
                                st.number_input(
                                    f"Page (1-{total_pages}):", 
                                    min_value=1, 
                                    max_value=total_pages, 
                                    value=st.session_state.current_page,
                                    key="current_page",
                                    on_change=page_change_callback
                                )
                            
                            with col3:
                                if st.session_state.current_page < total_pages:
                                    if st.button("Next Page →"):
                                        st.session_state.current_page += 1
                                        st.experimental_rerun()
                        else:
                            st.session_state.current_page = 1
                        
                        start_idx = (st.session_state.current_page - 1) * items_per_page
                        end_idx = min(start_idx + items_per_page, len(display_df))
                        
                        for idx, row in display_df.iloc[start_idx:end_idx].iterrows():
                            headline = row['headline'] if pd.notna(row['headline']) else "No headline available"
                            url = row['article_url']
                            domain = get_domain(url)
                            if st.session_state.search_term and st.session_state.search_term.lower() in headline.lower():
                                pattern = re.compile(re.escape(st.session_state.search_term), re.IGNORECASE)
                                headline = pattern.sub(f"<span class='highlight'>{st.session_state.search_term}</span>", headline)
                            
                            st.markdown(
                                f"""
                                <div class="article-card">
                                    <div class="article-title">{headline}</div>
                                    <div class="article-url">
                                        <strong>Source:</strong> {domain}<br>
                                        <a href="{url}" target="_blank">{url}</a>
                                    </div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                else:
                    st.warning(f"No articles found in cluster {st.session_state.selected_cluster}")
        
        with tab3:
            st.header("Cluster Visualization")

            with st.spinner("Generating visualization... This may take a minute for large datasets."):
                vis_sample = df
                if len(df) > 5000:  # Limit sample size for performance
                    vis_sample = df.sample(5000, random_state=42)
                    st.info(f"Using a sample of 5000 articles out of {len(df)} for visualization.")
                vis_figure = visualize_clusters(vis_sample, kmeans, svd)
                
                if vis_figure:
                    st.plotly_chart(vis_figure, use_container_width=True)
                    st.markdown("""
                        This visualization shows how articles cluster in the feature space.
                        Articles that are close to each other are more similar in content.
                        
                        - Each point represents a news article
                        - Colors represent different categories
                        - Hover over points to see article headlines
                    """)
                else:
                    st.error("Unable to generate visualization.")
    else:
      
        st.info("Please load data using the sidebar options to begin exploring news clusters.")

if __name__ == "__main__":
    main()