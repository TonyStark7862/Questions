import pandas as pd
import time
import random
import uuid
import datetime
import os
import threading
import csv

# Simulated LLM API function - Replace with your actual API call
def abc_response(prompt):
    """Simulated LLM API call. Replace with your actual API implementation."""
    # In a real implementation, this would call your LLM API
    # For this demo, we'll create some sample responses
    responses = [
        f"This is an interesting topic! I think {random.choice(['it depends on context', 'we should consider all angles', 'there are multiple perspectives'])}.",
        f"I {random.choice(['agree', 'disagree', 'partially agree'])} with that. {random.choice(['Because...', 'My reasoning is...', 'I think...'])}",
        f"Great point! {random.choice(['Have you considered...?', 'I would add that...', 'Additionally...'])}",
        f"Thanks for sharing! {random.choice(['I learned something new.', 'That\'s fascinating.', 'I hadn\'t thought of it that way.'])}"
    ]
    return random.choice(responses)

# Initialize CSV files for agent system
def initialize_agent_system():
    # Create agents.csv if it doesn't exist
    if not os.path.exists("agents.csv"):
        agents = []
        # Generate 10 sample agents
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
            
            posting_frequency = random.choice(["high", "medium", "low"])
            
            agent = {
                'username': f"agent_{i}",
                'email': f"agent_{i}@example.com",
                'personality_traits': personality_traits,
                'interests': interests,
                'writing_style': writing_style,
                'posting_frequency': posting_frequency,
                'last_active': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            agents.append(agent)
        
        # Save agents to CSV
        pd.DataFrame(agents).to_csv("agents.csv", index=False)
        print("Created agents.csv with 10 sample agents")

    # Create agent_activities.csv if it doesn't exist
    if not os.path.exists("agent_activities.csv"):
        pd.DataFrame(columns=[
            'activity_id', 'agent_username', 'activity_type', 'content_id', 'timestamp'
        ]).to_csv("agent_activities.csv", index=False)
        print("Created agent_activities.csv")

    # Create agent_interactions.csv if it doesn't exist
    if not os.path.exists("agent_interactions.csv"):
        pd.DataFrame(columns=[
            'interaction_id', 'agent_username', 'real_user', 'content_id', 'interaction_type', 'timestamp'
        ]).to_csv("agent_interactions.csv", index=False)
        print("Created agent_interactions.csv")

    # Create usage_stats.csv if it doesn't exist
    if not os.path.exists("usage_stats.csv"):
        pd.DataFrame(columns=[
            'timestamp', 'calls_made', 'tokens_used', 'cost'
        ]).to_csv("usage_stats.csv", index=False)
        print("Created usage_stats.csv")
    
    # Create sample images folder and add some images (for demo purposes)
    if not os.path.exists("images"):
        os.makedirs("images")
        # In a real implementation, you would add actual images here
        print("Created images directory")

# Log agent activity
def log_activity(agent_username, activity_type, content_id):
    new_activity = {
        'activity_id': str(uuid.uuid4()),
        'agent_username': agent_username,
        'activity_type': activity_type,
        'content_id': content_id,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Append to CSV
    with open("agent_activities.csv", 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_activity.keys())
        writer.writerow(new_activity)

# Log LLM API usage
def log_api_usage(calls=1, tokens=100, cost=0.002):
    new_usage = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'calls_made': calls,
        'tokens_used': tokens,
        'cost': cost
    }
    
    # Append to CSV
    with open("usage_stats.csv", 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_usage.keys())
        writer.writerow(new_usage)

# Create a post from an agent
def create_agent_post(agent):
    # Generate post content using LLM
    prompt = f"""
    Create a social media post from a person with these characteristics:
    - Personality: {agent['personality_traits']}
    - Interests: {agent['interests']}
    - Writing style: {agent['writing_style']}
    Keep it brief and conversational.
    """
    
    post_content = abc_response(prompt)
    
    # Log API usage
    log_api_usage()
    
    # Create post entry
    post_id = str(uuid.uuid4())
    new_post = {
        'post_id': post_id,
        'author': agent['username'],
        'content': post_content,
        'image_path': "",  # No image for simplicity
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Append to posts.csv
    posts = pd.read_csv("posts.csv")
    posts = pd.concat([posts, pd.DataFrame([new_post])], ignore_index=True)
    posts.to_csv("posts.csv", index=False)
    
    # Log activity
    log_activity(agent['username'], 'post', post_id)
    
    print(f"Agent {agent['username']} created a post: {post_content[:30]}...")
    return post_id

# Create a comment from an agent on a post
def create_agent_comment(agent, post_id):
    # Read posts to get post content
    posts = pd.read_csv("posts.csv")
    post = posts[posts['post_id'] == post_id].iloc[0]
    
    # Generate comment content using LLM
    prompt = f"""
    You are a person with these characteristics:
    - Personality: {agent['personality_traits']}
    - Interests: {agent['interests']}
    - Writing style: {agent['writing_style']}
    
    Create a brief comment responding to this post:
    "{post['content']}"
    """
    
    comment_content = abc_response(prompt)
    
    # Log API usage
    log_api_usage()
    
    # Create comment entry
    comment_id = str(uuid.uuid4())
    new_comment = {
        'comment_id': comment_id,
        'post_id': post_id,
        'author': agent['username'],
        'content': comment_content,
        'parent_comment_id': None,  # Top-level comment
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Append to comments.csv
    comments = pd.read_csv("comments.csv")
    comments = pd.concat([comments, pd.DataFrame([new_comment])], ignore_index=True)
    comments.to_csv("comments.csv", index=False)
    
    # Log activity
    log_activity(agent['username'], 'comment', comment_id)
    
    print(f"Agent {agent['username']} commented on post: {comment_content[:30]}...")
    return comment_id

# Create a reply from an agent to a comment
def create_agent_reply(agent, comment_id):
    # Read comments to get comment content
    comments = pd.read_csv("comments.csv")
    comment = comments[comments['comment_id'] == comment_id].iloc[0]
    
    # Generate reply content using LLM
    prompt = f"""
    You are a person with these characteristics:
    - Personality: {agent['personality_traits']}
    - Interests: {agent['interests']}
    - Writing style: {agent['writing_style']}
    
    Create a brief reply to this comment:
    "{comment['content']}"
    """
    
    reply_content = abc_response(prompt)
    
    # Log API usage
    log_api_usage()
    
    # Create reply entry
    reply_id = str(uuid.uuid4())
    new_reply = {
        'comment_id': reply_id,
        'post_id': comment['post_id'],
        'author': agent['username'],
        'content': reply_content,
        'parent_comment_id': comment_id,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Append to comments.csv
    comments = pd.concat([comments, pd.DataFrame([new_reply])], ignore_index=True)
    comments.to_csv("comments.csv", index=False)
    
    # Log activity
    log_activity(agent['username'], 'reply', reply_id)
    
    print(f"Agent {agent['username']} replied to comment: {reply_content[:30]}...")
    return reply_id

# Check for new content from real users
def check_for_real_user_activity():
    try:
        # Load agents
        agents = pd.read_csv("agents.csv")
        agent_usernames = agents['username'].tolist()
        
        # Load posts
        posts = pd.read_csv("posts.csv")
        comments = pd.read_csv("comments.csv")
        
        # Check for posts by real users (not in agent_usernames)
        real_user_posts = posts[~posts['author'].isin(agent_usernames)]
        
        # Get recent posts (last 5 minutes)
        current_time = datetime.datetime.now()
        five_minutes_ago = (current_time - datetime.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        recent_real_posts = real_user_posts[real_user_posts['timestamp'] > five_minutes_ago]
        
        # Respond to recent real user posts
        if not recent_real_posts.empty:
            for _, post in recent_real_posts.iterrows():
                # Check if any agent has already commented
                post_comments = comments[comments['post_id'] == post['post_id']]
                agent_comments = post_comments[post_comments['author'].isin(agent_usernames)]
                
                if agent_comments.empty:
                    # Choose a random agent to respond
                    agent = agents.sample(1).iloc[0]
                    create_agent_comment(agent, post['post_id'])
        
        # Check for comments by real users
        real_user_comments = comments[~comments['author'].isin(agent_usernames)]
        recent_real_comments = real_user_comments[real_user_comments['timestamp'] > five_minutes_ago]
        
        # Respond to recent real user comments
        if not recent_real_comments.empty:
            for _, comment in recent_real_comments.iterrows():
                # Check if this comment is on a post by an agent
                post_id = comment['post_id']
                post_author = posts[posts['post_id'] == post_id]['author'].iloc[0]
                
                if post_author in agent_usernames:
                    # The agent who made the post should reply
                    agent = agents[agents['username'] == post_author].iloc[0]
                    create_agent_reply(agent, comment['comment_id'])
                else:
                    # Check if comment is a reply to an agent's comment
                    if not pd.isna(comment['parent_comment_id']):
                        parent_id = comment['parent_comment_id']
                        parent_author = comments[comments['comment_id'] == parent_id]['author'].iloc[0]
                        
                        if parent_author in agent_usernames:
                            # The agent who was replied to should respond
                            agent = agents[agents['username'] == parent_author].iloc[0]
                            create_agent_reply(agent, comment['comment_id'])
        
    except Exception as e:
        print(f"Error checking for real user activity: {e}")

# Schedule random agent activities
def schedule_agent_activities():
    try:
        # Load agents
        agents = pd.read_csv("agents.csv")
        
        # Load existing content
        posts = pd.read_csv("posts.csv")
        comments = pd.read_csv("comments.csv")
        
        # Random chance for an agent to post
        if random.random() < 0.3:  # 30% chance
            # Choose a random agent
            agent = agents.sample(1).iloc[0]
            create_agent_post(agent)
        
        # Random chance for an agent to comment on a post
        if not posts.empty and random.random() < 0.5:  # 50% chance
            # Choose a random agent
            agent = agents.sample(1).iloc[0]
            # Choose a random post
            post = posts.sample(1).iloc[0]
            create_agent_comment(agent, post['post_id'])
        
        # Random chance for an agent to reply to a comment
        if not comments.empty and random.random() < 0.3:  # 30% chance
            # Choose a random agent
            agent = agents.sample(1).iloc[0]
            # Choose a random comment (not a reply)
            top_comments = comments[comments['parent_comment_id'].isna()]
            if not top_comments.empty:
                comment = top_comments.sample(1).iloc[0]
                create_agent_reply(agent, comment['comment_id'])
    
    except Exception as e:
        print(f"Error scheduling agent activities: {e}")

# Main agent loop
def run_agent_system():
    print("Starting AI Agent System...")
    
    # Initialize agent system
    initialize_agent_system()
    
    # Main loop
    while True:
        try:
            # Check for real user activity
            check_for_real_user_activity()
            
            # Schedule random agent activities
            schedule_agent_activities()
            
            # Sleep for a random time (5-15 seconds)
            sleep_time = random.randint(5, 15)
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error in agent system: {e}")
            # Sleep a bit before retrying
            time.sleep(5)

# Run the agent system in a separate thread
def start_agent_system():
    agent_thread = threading.Thread(target=run_agent_system)
    agent_thread.daemon = True  # Thread will exit when main program exits
    agent_thread.start()
    return agent_thread

# Entry point
if __name__ == "__main__":
    # Start the agent system
    agent_thread = start_agent_system()
    
    print("Agent system is running in the background.")
    print("You can run your Streamlit app separately with:")
    print("streamlit run app.py")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agent system...")
