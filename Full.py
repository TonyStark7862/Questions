import streamlit as st
import pandas as pd
import os
import datetime
import uuid
import random
import time
import threading
from PIL import Image

# Initialize CSV files
def initialize_files():
    if not os.path.exists("images"):
        os.makedirs("images")

    if not os.path.exists("posts.csv"):
        pd.DataFrame(columns=['post_id', 'author', 'content', 'image_path', 'timestamp']).to_csv("posts.csv", index=False)

    if not os.path.exists("comments.csv"):
        pd.DataFrame(columns=['comment_id', 'post_id', 'author', 'content', 'parent_comment_id', 'timestamp']).to_csv("comments.csv", index=False)
        
    if not os.path.exists("agents.csv"):
        agents = []
        for i in range(10):
            personality_traits = random.choice([
                "friendly and outgoing",
                "thoughtful and analytical",
                "humorous and witty",
                "direct and straightforward",
                "creative and imaginative"
            ])
            
            interests = random.choice([
                "technology, science, space",
                "arts, music, culture",
                "sports, fitness, health",
                "politics, current events, history",
                "food, travel, lifestyle"
            ])
            
            writing_style = random.choice([
                "casual and conversational",
                "formal and structured",
                "concise and clear",
                "detailed and elaborate",
                "quirky and unique"
            ])
            
            agent = {
                'username': f"agent_{i}",
                'email': f"agent_{i}@example.com",
                'personality_traits': personality_traits,
                'interests': interests,
                'writing_style': writing_style,
                'posting_frequency': random.choice(["high", "medium", "low"]),
                'last_active': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            agents.append(agent)
        
        pd.DataFrame(agents).to_csv("agents.csv", index=False)

# Simulated LLM API function
def abc_response(prompt):
    responses = [
        f"This is interesting! I think {random.choice(['we need more context', 'this has multiple perspectives', 'this is worth exploring further'])}.",
        f"I {random.choice(['agree completely', 'somewhat agree', 'have mixed feelings about this'])}. {random.choice(['My experience suggests...', 'From what I understand...', 'I believe...'])}",
        f"Great point! {random.choice(['Have you considered...?', 'I would add that...', 'This reminds me of...'])}",
        f"Thanks for sharing this! {random.choice(['I appreciate your perspective.', 'That\'s a fascinating take.', 'I hadn\'t thought about it that way before.'])}"
    ]
    return random.choice(responses)

# Agent functions
def create_agent_post():
    try:
        # Load agents
        agents = pd.read_csv("agents.csv")
        agent = agents.sample(1).iloc[0]
        
        # Generate post content
        prompt = "Create a social media post about " + agent['interests']
        post_content = abc_response(prompt)
        
        # Create post
        post_id = str(uuid.uuid4())
        new_post = {
            'post_id': post_id,
            'author': agent['username'],
            'content': post_content,
            'image_path': "",
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save post
        posts = pd.read_csv("posts.csv")
        posts = pd.concat([posts, pd.DataFrame([new_post])], ignore_index=True)
        posts.to_csv("posts.csv", index=False)
        
        return post_id
    except Exception as e:
        print(f"Error creating agent post: {e}")
        return None

def create_agent_comment(post_id):
    try:
        # Load agents and posts
        agents = pd.read_csv("agents.csv")
        posts = pd.read_csv("posts.csv")
        
        # Find the post
        post = posts[posts['post_id'] == post_id]
        if post.empty:
            return None
            
        post = post.iloc[0]
        
        # Choose a random agent
        agent = agents.sample(1).iloc[0]
        
        # Generate comment
        prompt = "Comment on this post: " + post['content']
        comment_content = abc_response(prompt)
        
        # Create comment
        comment_id = str(uuid.uuid4())
        new_comment = {
            'comment_id': comment_id,
            'post_id': post_id,
            'author': agent['username'],
            'content': comment_content,
            'parent_comment_id': None,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save comment
        comments = pd.read_csv("comments.csv")
        comments = pd.concat([comments, pd.DataFrame([new_comment])], ignore_index=True)
        comments.to_csv("comments.csv", index=False)
        
        return comment_id
    except Exception as e:
        print(f"Error creating agent comment: {e}")
        return None

def create_agent_reply(comment_id):
    try:
        # Load agents and comments
        agents = pd.read_csv("agents.csv")
        comments = pd.read_csv("comments.csv")
        
        # Find the comment
        comment = comments[comments['comment_id'] == comment_id]
        if comment.empty:
            return None
            
        comment = comment.iloc[0]
        
        # Choose a random agent
        agent = agents.sample(1).iloc[0]
        
        # Generate reply
        prompt = "Reply to this comment: " + comment['content']
        reply_content = abc_response(prompt)
        
        # Create reply
        reply_id = str(uuid.uuid4())
        new_reply = {
            'comment_id': reply_id,
            'post_id': comment['post_id'],
            'author': agent['username'],
            'content': reply_content,
            'parent_comment_id': comment['comment_id'],
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save reply
        comments = pd.concat([comments, pd.DataFrame([new_reply])], ignore_index=True)
        comments.to_csv("comments.csv", index=False)
        
        return reply_id
    except Exception as e:
        print(f"Error creating agent reply: {e}")
        return None

# Function to check for new real user activity and respond
def agent_activity_cycle():
    while True:
        try:
            # Create a random post from an agent (20% chance)
            if random.random() < 0.2:
                create_agent_post()
            
            # Respond to real user posts
            posts = pd.read_csv("posts.csv")
            comments = pd.read_csv("comments.csv")
            agents = pd.read_csv("agents.csv")
            
            # Get agent usernames
            agent_usernames = agents['username'].tolist()
            
            # Find real user posts that don't have agent comments
            real_user_posts = posts[~posts['author'].isin(agent_usernames)]
            
            for _, post in real_user_posts.iterrows():
                # Check if post has agent comments
                post_comments = comments[comments['post_id'] == post['post_id']]
                agent_comments = post_comments[post_comments['author'].isin(agent_usernames)]
                
                # If fewer than 2 agent comments, add one
                if len(agent_comments) < 2:
                    create_agent_comment(post['post_id'])
                    
            # Find real user comments that don't have agent replies
            real_user_comments = comments[~comments['author'].isin(agent_usernames)]
            
            for _, comment in real_user_comments.iterrows():
                # Check if comment has agent replies
                comment_replies = comments[comments['parent_comment_id'] == comment['comment_id']]
                agent_replies = comment_replies[comment_replies['author'].isin(agent_usernames)]
                
                # If no agent replies, add one (50% chance)
                if len(agent_replies) == 0 and random.random() < 0.5:
                    create_agent_reply(comment['comment_id'])
            
            # Sleep for a short time before next cycle
            time.sleep(5)
            
        except Exception as e:
            print(f"Error in agent activity cycle: {e}")
            time.sleep(5)

# Start agent thread
def start_agent_thread():
    agent_thread = threading.Thread(target=agent_activity_cycle)
    agent_thread.daemon = True
    agent_thread.start()
    return agent_thread

# Initialize session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
    st.session_state.agent_thread_started = False

# Initialize files
initialize_files()

# Start agent thread if not already started
if not st.session_state.agent_thread_started:
    start_agent_thread()
    st.session_state.agent_thread_started = True

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
        posts = pd.read_csv("posts.csv")
        
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
        
        # Have agents comment on this post immediately
        new_post_id = new_post['post_id']
        threading.Thread(target=lambda: create_agent_comment(new_post_id)).start()
        
        st.rerun()

# Load data
@st.cache_data(ttl=2)  # Cache for 2 seconds
def load_data():
    posts = pd.read_csv("posts.csv")
    comments = pd.read_csv("comments.csv")
    return posts, comments

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
                    
                    # Add reply form for each comment (not in an expander to avoid nesting issues)
                    reply_key = f"reply_{comment['comment_id']}"
                    reply_content = st.text_area("Write a reply", key=reply_key, label_visibility="collapsed")
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
                            
                            # Have an agent reply to this comment
                            threading.Thread(target=lambda comment_id=new_reply['comment_id']: 
                                create_agent_reply(comment_id)).start()
                                
                            st.rerun()
                    
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
                        
                        # Have an agent reply to this comment
                        threading.Thread(target=lambda comment_id=new_comment_data['comment_id']: 
                            create_agent_reply(comment_id)).start()
                            
                        st.rerun()
            
            st.markdown("---")
else:
    st.info("No posts yet. Be the first to post something!")
    
    # Create some initial agent posts
    if random.random() < 0.8:  # 80% chance
        post_id = create_agent_post()
        if post_id:
            st.rerun()

# Add some instructions
with st.sidebar:
    st.header("How to use")
    st.write("1. Enter your name")
    st.write("2. Create a post")
    st.write("3. Comment on posts")
    st.write("4. Reply to comments")
    
    st.header("About")
    st.write("This is a simple social media app with AI users.")
    
    # Force refresh button
    if st.button("Refresh Feed"):
        st.rerun()
