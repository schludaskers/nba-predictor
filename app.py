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

# --- CUSTOM CSS (THE VISUAL UPGRADE) ---
st.markdown("""
<style>
    /* Dark Background & Text */
    .stApp {
        background-color: #0F1116;
        color: #FFFFFF;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 30, 40, 0.7);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(0, 255, 255, 0.3);
    }

    /* Typography */
    .big-stat {
        font-size: 36px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00ADB5, #00FFF5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 14px;
        font-weight: 600;
        color: #8899AC;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .player-name {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .team-name {
        font-size: 16px;
        color: #8899AC;
    }

    /* Betting Badges */
    .badge-over {
        background-color: rgba(0, 255, 127, 0.2);
        color: #00FF7F;
        padding: 5px 10px;
        border-radius: 8px;
        font-weight: bold;
        border: 1px solid #00FF7F;
    }
    .badge-under {
        background-color: rgba(255, 46, 99, 0.2);
        color: #FF2E63;
        padding: 5px 10px;
        border-radius: 8px;
        font-weight: bold;
        border: 1px solid #FF2E63;
    }
    
    /* Input Styling */
    .stNumberInput input {
        background-color: #1E1E28;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- TRAINING ENGINE ---
@st.cache_resource
def train_model_from_csv():
    try:
        df = pd.read_csv('nba_training_data.csv')
        
        # Verify columns exist
        needed = ['PTS_L5', 'REB_L5', 'AST_L5', 'MIN_L5', 'STL_L5', 'BLK_L5']
        if not all(col in df.columns for col in needed):
            return None
            
        targets = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        models = {}
        
        for target in targets:
            if target in df.columns:
                X = df[needed]
                y = df[target]
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                model.fit(X, y)
                models[target] = model
        return models
    except FileNotFoundError:
        return None

# --- HELPERS ---
def get_headshot_url(player_id):
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

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
        preds[stat] = round(model.predict(feature_row)[0], 1)
    return preds

# --- MAIN UI ---
models = train_model_from_csv()
if not models:
    st.error("⚠️ Data file not found. Please upload 'nba_training_data.csv'")
    st.stop()

# Sidebar
st.sidebar.markdown("### 📅 Schedule Settings")
selected_date = st.sidebar.date_input("Game Date", datetime.now())

# Check Schedule
with st.sidebar:
    with st.spinner("Loading schedule..."):
        games_df = get_todays_games(selected_date)
        if not games_df.empty:
            st.success(f"{len(games_df)} Games Today")
            st.dataframe(games_df[['HOME_TEAM_ID', 'VISITOR_TEAM_ID']], hide_index=True)
        else:
            st.warning("No games scheduled.")

# Main Header
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🏀 NBA PROP ASSASSIN</h1>", unsafe_allow_html=True)

# Search Bar (Centered)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    player_name = st.text_input("", placeholder="🔍 Search Player (e.g. LeBron James)...", label_visibility="collapsed")
    search_btn = st.button("Analyze Player", use_container_width=True, type="primary")

if search_btn and player_name:
    active_players = players.get_active_players()
    player_info = next((p for p in active_players if p['full_name'].lower() == player_name.lower()), None)
    
    if player_info:
        # Fetch Data
        recent_df = get_player_recent_stats(player_info['id'])
        
        if not recent_df.empty and len(recent_df) >= 5:
            preds = predict(models, recent_df)
            
            # --- PLAYER PROFILE HEADER ---
            with st.container():
                st.markdown(f"""
                <div class='glass-card' style='display: flex; align-items: center; gap: 20px;'>
                    <img src='{get_headshot_url(player_info['id'])}' style='border-radius: 10px; height: 120px;'>
                    <div>
                        <div class='player-name'>{player_info['full_name']}</div>
                        <div class='team-name'>ID: {player_info['id']} • 2025-26 Season</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- PROP COMPARATOR TOOL ---
            st.markdown("### 📊 Prop Betting Analyzer")
            
            # Create tabs for different stats
            tab_pts, tab_reb, tab_ast = st.tabs(["Points", "Rebounds", "Assists"])
            
            def render_betting_card(stat_name, pred_val, recent_logs, stat_col):
                c_metrics, c_chart = st.columns([1, 2])
                
                with c_metrics:
                    st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='stat-label'>AI Prediction</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='big-stat'>{pred_val}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # User Input for Line
                    line = st.number_input(f"Enter {stat_name} Line:", value=float(int(pred_val)), step=0.5, key=f"line_{stat_name}")
                    
                    diff = pred_val - line
                    
                    if diff > 0:
                        st.markdown(f"<div style='margin-top: 10px;'><span class='badge-over'>BET OVER</span> (+{diff:.1f} edge)</div>", unsafe_allow_html=True)
                    elif diff < 0:
                        st.markdown(f"<div style='margin-top: 10px;'><span class='badge-under'>BET UNDER</span> ({diff:.1f} edge)</div>", unsafe_allow_html=True)
                    else:
                        st.write("No Edge (Push)")
                        
                    st.markdown("</div>", unsafe_allow_html=True)

                with c_chart:
                    # Trendy Chart
                    fig = go.Figure()
                    last_10 = recent_logs.tail(10) # Show last 10 for better context
                    
                    # Area Chart
                    fig.add_trace(go.Scatter(
                        x=last_10['GAME_DATE'], 
                        y=last_10[stat_col],
                        fill='tozeroy',
                        mode='lines+markers',
                        line=dict(color='#00ADB5', width=3),
                        name='Actual',
                        fillcolor='rgba(0, 173, 181, 0.2)'
                    ))
                    
                    # The Line (User Input)
                    fig.add_hline(y=line, line_dash="dash", line_color="#FFFFFF", annotation_text="The Line")
                    # The Prediction
                    fig.add_hline(y=pred_val, line_dash="dot", line_color="#00ADB5", annotation_text="AI Pred")
                    
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=300,
                        title=f"Last 10 Games vs Line ({line})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab_pts:
                render_betting_card("Points", preds['PTS'], recent_df, 'PTS')
            with tab_reb:
                render_betting_card("Rebounds", preds['REB'], recent_df, 'REB')
            with tab_ast:
                render_betting_card("Assists", preds['AST'], recent_df, 'AST')

        else:
            st.error("Not enough recent data (need at least 5 games) or player inactive.")
    else:
        st.error("Player not found. Check spelling.")
