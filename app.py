import streamlit as st
import pandas as pd
import altair as alt
import json
import sqlite3
import hashlib
import os

st.set_page_config(
    page_title="RL-Vis-Lite",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 400px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DB_FILE = "rl_vis_lite.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            run_name TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def process_data(df, file_name):
    if 'value_estimate' in df.columns:
        df.rename(columns={'value_estimate': 'confidence_metric'}, inplace=True)
    elif 'max_q_value' in df.columns:
        df.rename(columns={'max_q_value': 'confidence_metric'}, inplace=True)
    
    df['file_label'] = file_name
    
    required_cols = ['timestep', 'cumulative_reward', 'confidence_metric']
    if not all(col in df.columns for col in required_cols):
        return None, f"File '{file_name}' is missing required columns."
    
    return df, None

def downsample_visual_data(df, sample_rate):
    def sample_group(group):
        reward_data = group.dropna(subset=['cumulative_reward'])
        no_reward_mask = group['cumulative_reward'].isna()
        sampled_confidence_data = group[no_reward_mask].iloc[::sample_rate, :]
        return pd.concat([reward_data, sampled_confidence_data])

    sampled_df = df.groupby('file_label').apply(sample_group).reset_index(drop=True)
    return sampled_df.sort_values(by='timestep')


def render_login_page():
    st.title("Welcome to RL-Vis-Lite")
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    
    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_button", use_container_width=True):
            c.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
            user_data = c.fetchone()
            if user_data and user_data[1] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.user_id = user_data[0]
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Invalid email or password")
                
    with signup_tab:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Sign Up", key="signup_button", use_container_width=True):
            if not email or not password:
                st.error("Email and password cannot be empty")
            else:
                try:
                    c.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, hash_password(password)))
                    conn.commit()
                    st.session_state.logged_in = True
                    st.session_state.user_id = c.lastrowid
                    st.session_state.user_email = email
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Email already exists.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")
    conn.close()

def render_dashboard_page():
    user_id = st.session_state.user_id
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'current_run_name' not in st.session_state:
        st.session_state.current_run_name = "New Run"

    st.sidebar.title("RL-Vis-Lite")
    st.sidebar.write(f"Welcome, **{st.session_state.user_email}**")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.session_state.logged_in = False
        st.rerun()
    
    st.sidebar.divider()
    
    with st.sidebar.expander("Upload New Run", expanded=True):
        new_files = st.file_uploader("Upload one or more .csv log files", type=['csv'], accept_multiple_files=True)
        new_run_name = st.text_input("Enter a name for this run")
        
        if st.button("Process & Save Run", use_container_width=True):
            if new_files and new_run_name:
                with st.spinner("Processing and saving run..."):
                    all_dfs = []
                    has_error = False
                    for file in new_files:
                        df = pd.read_csv(file)
                        processed_df, error = process_data(df, file.name)
                        if error:
                            st.error(error)
                            has_error = True
                            break
                        all_dfs.append(processed_df)
                    
                    if not has_error:
                        combined_df = pd.concat(all_dfs, ignore_index=True)
                        data_json = combined_df.to_json(orient='records')
                        created_at = pd.Timestamp.now().isoformat()
                        c.execute("INSERT INTO runs (user_id, run_name, data_json, created_at) VALUES (?, ?, ?, ?)",
                                  (user_id, new_run_name, data_json, created_at))
                        conn.commit()
                        st.session_state.current_df = combined_df
                        st.session_state.current_run_name = new_run_name
                        st.success(f"Run '{new_run_name}' saved!")
                        st.rerun()
            else:
                st.warning("Please upload at least one file and provide a name.")

    st.sidebar.divider()
    
    st.sidebar.subheader("My Saved Runs")
    c.execute("SELECT id, run_name, created_at FROM runs WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    runs = c.fetchall()
    
    if not runs:
        st.sidebar.info("No saved runs. Upload one to get started!")
    else:
        for run_id, run_name, created_at in runs:
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            with col1:
                if st.button(run_name, key=f"load_{run_id}", use_container_width=True):
                    with st.spinner(f"Loading '{run_name}'..."):
                        c.execute("SELECT data_json FROM runs WHERE id = ?", (run_id,))
                        data_json = c.fetchone()[0]
                        st.session_state.current_df = pd.read_json(data_json, orient='records')
                        st.session_state.current_run_name = run_name
                        st.rerun()
            with col2:
                if st.button("X", key=f"del_{run_id}", use_container_width=True, help="Delete run"):
                    c.execute("DELETE FROM runs WHERE id = ?", (run_id,))
                    conn.commit()
                    st.rerun()


    st.title(st.session_state.current_run_name)
    
    if st.session_state.current_df is None:
        st.info("Upload a new run or select a saved run from the sidebar to begin analysis.")
        conn.close()
        return

    try:
        df = st.session_state.current_df
        
        st.header("Key Performance Indicators (KPIs)")
        
        grouped_data = df.groupby('file_label')
        num_runs = len(grouped_data)
        kpi_cols = st.columns(num_runs)
        
        if num_runs == 1:
            kpi_cols = st.columns(3)
        
        run_counter = 0
        for file_label, data in grouped_data:
            col = kpi_cols[run_counter % len(kpi_cols)]
            with col:
                st.subheader(f"File: {file_label}")
                reward_data_full = data.dropna(subset=['cumulative_reward'])
                
                total_timesteps = data['timestep'].max()
                max_reward = reward_data_full['cumulative_reward'].max()
                
                avg_reward_last_100 = 0
                if not reward_data_full.empty:
                    if len(reward_data_full) >= 100:
                        avg_reward_last_100 = reward_data_full['cumulative_reward'].tail(100).mean()
                    else:
                        avg_reward_last_100 = reward_data_full['cumulative_reward'].mean()

                st.metric(label="Total Timesteps", value=f"{total_timesteps:,}")
                st.metric(label="Max Reward Achieved", value=f"{max_reward:,.2f}")
                st.metric(label="Avg. Reward (Last 100)", value=f"{avg_reward_last_100:,.2f}")
            run_counter += 1

        st.divider()

        st.header("Visualization Controls")
        
        sample_rate = st.slider(
            "Adjust Plot Detail (Sample Rate)", 
            min_value=1, 
            max_value=5000, 
            value=100, 
            step=200,
            help="To improve performance, we only plot a sample of the confidence data. A higher rate means fewer points are plotted (faster). Set to 1 for full detail."
        )
        
        sampled_df = downsample_visual_data(df, sample_rate)
        st.write(f"Displaying {len(sampled_df)} of {len(df)} total data points for charts.")
        
        st.divider()

        st.header("Analyze Training Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Agent Performance (Reward)")
            smooth_rewards = st.checkbox("Smooth Reward Chart (Moving Average)")
            window_size = 10
            if smooth_rewards:
                window_size = st.slider(
                    "Smoothing Window Size", 
                    min_value=10, 
                    max_value=200, 
                    value=50, 
                    step=10,
                    help="The number of episodes to average over."
                )
            
            reward_data = sampled_df.dropna(subset=['cumulative_reward'])
            
            if not reward_data.empty:
                y_axis = 'cumulative_reward'
                if smooth_rewards:
                    reward_data['smoothed_reward'] = reward_data.groupby('file_label')['cumulative_reward'].rolling(window=size, min_periods=1).mean().reset_index(0, drop=True)
                    y_axis = 'smoothed_reward'
                
                chart = alt.Chart(reward_data).mark_line().encode(
                    x=alt.X('timestep', title='Timestep'),
                    y=alt.Y(y_axis, title='Reward'),
                    color=alt.Color('file_label', title='File'),
                    tooltip=['timestep', y_axis, 'file_label']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No episode rewards found in this log.")

        with col2:
            st.subheader("Agent Confidence (confidence_metric)")
            chart = alt.Chart(sampled_df).mark_line().encode(
                x=alt.X('timestep', title='Timestep'),
                y=alt.Y('confidence_metric', title='Confidence'),
                color=alt.Color('file_label', title='File'),
                tooltip=['timestep', 'confidence_metric', 'file_label']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while displaying charts: {e}")
        st.exception(e)
    
    conn.close()

def main():
    init_db()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_email = None

    if st.session_state.logged_in:
        render_dashboard_page()
    else:
        render_login_page()

if __name__ == "__main__":
    main()


# streamlit run app.py