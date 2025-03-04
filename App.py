#streamlit-social-app.py

import streamlit as st
import pandas as pd
import os
import datetime
import uuid
import random
from PIL import Image

# Initialize session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# Create directories and CSV files if they don't exist
if not os.path.exists("images"):
    os.makedirs("images")

# Initialize CSV files if they don't exist
if not os.path.exists("posts.csv"):
    pd.DataFrame(columns=['post_id', 'author', 'content', 'image_path', 'timestamp']).to_csv("posts.csv", index=False)

if not os.path.exists("comments.csv"):
    pd.DataFrame(columns=['comment_id', 'post_id', 'author', 'content', 'parent_comment_id', 'timestamp']).to_csv("comments.csv", index=False)

# Load data
@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_data():
    posts = pd.read_csv("posts.csv")
    comments = pd.read_csv("comments.csv")
    return posts, comments

# App title
st.title("Simple Social Media")

# User identification
user_name = st.text_input("Your Name", value=st.session_state.user_name)
st.session_state.user_name = user_name

# Post creation form
with st.form(key="post_form"):
    post_content = st.text_area("Create a post", height=100)
    
    # Simple image selection
    sample_images = os.listdir("images") if os.path.exists("images") and os.listdir("images") else []
    image_path = st.selectbox("Select an image (optional)", ["None"] + sample_images)
    
    submit_post = st.form_submit_button("Post")
    
    if submit_post and user_name and post_content:
        posts, _ = load_data()
        
        new_post = {
            'post_id': str(uuid.uuid4()),
            'author': user_name,
            'content': post_content,
            'image_path': image_path if image_path != "None" else "",
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        posts = pd.concat([posts, pd.DataFrame([new_post])], ignore_index=True)
        posts.to_csv("posts.csv", index=False)
        st.success("Post created!")
        st.experimental_rerun()

# Load and display posts
posts, comments = load_data()

if not posts.empty:
    # Sort posts by timestamp (newest first)
    posts = posts.sort_values(by='timestamp', ascending=False)
    
    for _, post in posts.iterrows():
        with st.container():
            st.subheader(f"{post['author']}")
            st.write(post['content'])
            
            # Display image if exists
            if post['image_path'] and post['image_path'] != "None" and os.path.exists(f"images/{post['image_path']}"):
                try:
                    image = Image.open(f"images/{post['image_path']}")
                    st.image(image, caption=post['image_path'], use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            
            st.caption(f"Posted on {post['timestamp']}")
            
            # Comment form
            with st.expander("Comments", expanded=True):
                # Filter comments for this post
                post_comments = comments[comments['post_id'] == post['post_id']]
                
                # Show top-level comments first
                top_comments = post_comments[post_comments['parent_comment_id'].isna()]
                
                for _, comment in top_comments.iterrows():
                    st.markdown(f"**{comment['author']}**: {comment['content']}")
                    st.caption(f"{comment['timestamp']}")
                    
                    # Add reply form for each comment
                    with st.expander("Reply", expanded=False):
                        reply_key = f"reply_{comment['comment_id']}"
                        reply_content = st.text_area("Write a reply", key=reply_key)
                        if st.button("Submit Reply", key=f"btn_{reply_key}"):
                            if user_name and reply_content:
                                new_reply = {
                                    'comment_id': str(uuid.uuid4()),
                                    'post_id': post['post_id'],
                                    'author': user_name,
                                    'content': reply_content,
                                    'parent_comment_id': comment['comment_id'],
                                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                comments = pd.concat([comments, pd.DataFrame([new_reply])], ignore_index=True)
                                comments.to_csv("comments.csv", index=False)
                                st.success("Reply added!")
                                st.experimental_rerun()
                    
                    # Show replies to this comment
                    replies = post_comments[post_comments['parent_comment_id'] == comment['comment_id']]
                    for _, reply in replies.iterrows():
                        st.markdown(f"↪️ **{reply['author']}**: {reply['content']}")
                        st.caption(f"{reply['timestamp']}")
                
                # Add new comment form
                new_comment = st.text_area("Add a comment", key=f"comment_{post['post_id']}")
                if st.button("Comment", key=f"btn_comment_{post['post_id']}"):
                    if user_name and new_comment:
                        new_comment_data = {
                            'comment_id': str(uuid.uuid4()),
                            'post_id': post['post_id'],
                            'author': user_name,
                            'content': new_comment,
                            'parent_comment_id': None,
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        comments = pd.concat([comments, pd.DataFrame([new_comment_data])], ignore_index=True)
                        comments.to_csv("comments.csv", index=False)
                        st.success("Comment added!")
                        st.experimental_rerun()
            
            st.markdown("---")
else:
    st.info("No posts yet. Be the first to post something!")

# Add some instructions
with st.sidebar:
    st.header("How to use")
    st.write("1. Enter your name")
    st.write("2. Create a post")
    st.write("3. Comment on posts")
    st.write("4. Reply to comments")
    
    st.header("About")
    st.write("This is a simple social media app. Data is stored in CSV files.")
