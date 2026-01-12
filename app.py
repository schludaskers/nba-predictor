# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2, playergamelog
from nba_api.stats.static import players
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NBA Prop Assassin", layout="wide", page_icon="🏀")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .metric-card { 
        background-color: #262730; 
        border: 1px solid #41444b; 
        border-radius: 10px; 
        padding: 10px; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .big-number { font-size: 26px; font-weight: bold; color: #00ADB5; }
    .stat-label { font-size: 12px; color: #AAAAAA; text-transform: uppercase; letter-spacing: 1px;}
</style>
""", unsafe_allow_html=True)

# --- TRAINING ENGINE ---
@st.cache_resource
def train_model_from_csv():
    try:
        df = pd.read_csv('nba_training_data.csv')
        
        # ADDED STL AND BLK TO FEATURES & TARGETS
        features = ['PTS_L5', 'REB_L5', 'AST_L5', 'MIN_L5', 'STL_L5', 'BLK_L5']
        targets = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        
        models = {}
        for target in targets:
            X = df[features]
            y = df[target]
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X, y)
            models[target] = model
            
        return models
    except FileNotFoundError:
        return None

# --- API HELPERS ---
def get_todays_games(date_str):
    board = scoreboardv2.ScoreboardV2(game_date=date_str)
    return board.game_header.get_data_frame()

def get_player_recent_stats(player_id):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
        df = gamelog.get_data_frames()[0]
        if df.empty: return pd.DataFrame()
        
        df.columns = df.columns.str.upper()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
        df = df.sort_values(by='GAME_DATE')
        return df
    except:
        return pd.DataFrame()

def predict(models, recent_stats):
    if len(recent_stats) < 5: return None
    
    last_5 = recent_stats.tail(5)
    
    # Calculate features including STL/BLK
    feature_row = pd.DataFrame({
        'PTS_L5': [last_5['PTS'].mean()],
        'REB_L5': [last_5['REB'].mean()],
        'AST_L5': [last_5['AST'].mean()],
        'MIN_L5': [last_5['MIN'].mean()],
        'STL_L5': [last_5['STL'].mean()],
        'BLK_L5': [last_5['BLK'].mean()]
    })
    
    preds = {}
    for stat, model in models.items():
        # Round STL/BLK to 1 decimal place? Or 2? 
        # Usually props are like 0.5 or 1.5, so 1 decimal is good.
        preds[stat] = round(model.predict(feature_row)[0], 1)
    return preds

# --- MAIN UI ---
st.title("🏀 NBA Prop Predictor")

# 1. Train
models = train_model_from_csv()
if not models:
    st.error("⚠️ `nba_training_data.csv` not found.")
    st.stop()

# 2. Sidebar
st.sidebar.header("Settings")
selected_date = st.sidebar.date_input("Select Date", datetime.now())

# 3. Content
with st.spinner(f"Checking schedule..."):
    games_df = get_todays_games(selected_date)

if games_df.empty:
    st.info("No games scheduled.")
else:
    st.sidebar.success(f"{len(games_df)} Games Found")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Player")
        player_name = st.text_input("Player Name", placeholder="e.g. Victor Wembanyama")
        btn = st.button("Predict Stats")
        
        if btn and player_name:
            active_players = players.get_active_players()
            player_info = next((p for p in active_players if p['full_name'].lower() == player_name.lower()), None)
            
            if player_info:
                st.success(f"Selected: {player_info['full_name']}")
                with st.spinner("Crunching numbers..."):
                    recent_df = get_player_recent_stats(player_info['id'])
                    
                    if not recent_df.empty:
                        preds = predict(models, recent_df)
                        if preds:
                            # --- 5 COLUMN LAYOUT ---
                            cols = st.columns(5)
                            metrics = [
                                ('PTS', preds['PTS']),
                                ('REB', preds['REB']),
                                ('AST', preds['AST']),
                                ('STL', preds['STL']),
                                ('BLK', preds['BLK'])
                            ]
                            
                            for col, (label, val) in zip(cols, metrics):
                                col.markdown(f"""
                                <div class='metric-card'>
                                    <div class='big-number'>{val}</div>
                                    <div class='stat-label'>{label}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Graph
                            st.markdown("#### Last 5 Games Trend")
                            l5 = recent_df.tail(5)
                            fig = go.Figure()
                            
                            # Add lines for defensive stats? 
                            # Usually too cluttered. Let's stick to PTS/REB/AST for graph
                            # or make it selectable. For now, let's keep PTS as main.
                            fig.add_trace(go.Scatter(x=l5['GAME_DATE'], y=l5['PTS'], name='PTS', line=dict(color='#00ADB5', width=3)))
                            fig.add_trace(go.Scatter(x=l5['GAME_DATE'], y=l5['BLK'], name='BLK', line=dict(color='#EEEEEE', width=1, dash='dot')))
                            fig.add_trace(go.Scatter(x=l5['GAME_DATE'], y=l5['STL'], name='STL', line=dict(color='#FF2E63', width=1, dash='dot')))
                            
                            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Need at least 5 games history.")
                    else:
                        st.error("Stats unavailable.")
            else:
                st.error("Player not found.")

    with col2:
        st.info(f"Models trained on {datetime.now().year} data. predicting across 5 categories.")
