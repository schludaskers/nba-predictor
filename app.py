# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2, playergamelog, commonteamroster
from nba_api.stats.static import players, teams
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NBA Prop Assassin", layout="wide", page_icon="🏀")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0F1116; color: #FFFFFF; }
    
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
    }
    
    /* Roster Buttons */
    .stButton button {
        width: 100%;
        background-color: #1E1E28;
        border: 1px solid #333;
        color: #ddd;
        transition: all 0.2s;
    }
    .stButton button:hover {
        border-color: #00ADB5;
        color: #00ADB5;
    }

    /* Typography */
    .big-stat {
        font-size: 36px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00ADB5, #00FFF5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label { font-size: 14px; font-weight: 600; color: #8899AC; text-transform: uppercase; letter-spacing: 1px; }
    .player-name { font-size: 28px; font-weight: 700; margin-bottom: 5px; }
    .team-name { font-size: 16px; color: #8899AC; }
    
    /* Banner Stats */
    .banner-stat-box {
        text-align: center;
        padding: 0 15px;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .banner-stat-box:last-child { border-right: none; }
    .banner-val { font-size: 20px; font-weight: 700; color: #fff; }
    .banner-label { font-size: 10px; color: #aaa; text-transform: uppercase; }

    /* Betting Badges */
    .badge-over { background-color: rgba(0, 255, 127, 0.2); color: #00FF7F; padding: 5px 10px; border-radius: 8px; font-weight: bold; border: 1px solid #00FF7F; }
    .badge-under { background-color: rgba(255, 46, 99, 0.2); color: #FF2E63; padding: 5px 10px; border-radius: 8px; font-weight: bold; border: 1px solid #FF2E63; }
</style>
""", unsafe_allow_html=True)

# --- TRAINING ENGINE ---
@st.cache_resource
def train_model_from_csv():
    try:
        df = pd.read_csv('nba_training_data.csv')
        needed = ['PTS_L5', 'REB_L5', 'AST_L5', 'MIN_L5', 'STL_L5', 'BLK_L5']
        if not all(col in df.columns for col in needed): return None
        
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
    except FileNotFoundError: return None

# --- API HELPERS ---
@st.cache_data
def get_team_map():
    nba_teams = teams.get_teams()
    return {t['id']: t['abbreviation'] for t in nba_teams}

@st.cache_data
def get_roster(team_id):
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2025-26')
        return roster.get_data_frames()[0]
    except:
        return pd.DataFrame()

@st.cache_data
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
        'PTS_L5': [last_5['PTS'].mean()], 'REB_L5': [last_5['REB'].mean()],
        'AST_L5': [last_5['AST'].mean()], 'MIN_L5': [last_5['MIN'].mean()],
        'STL_L5': [last_5['STL'].mean()], 'BLK_L5': [last_5['BLK'].mean()]
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

# --- SIDEBAR (TEAMS & GAMES) ---
st.sidebar.markdown("### 📅 Settings")
selected_date = st.sidebar.date_input("Game Date", datetime.now())

# Get Games & Teams
team_list = teams.get_teams()
team_options = {t['full_name']: t['id'] for t in team_list}

with st.sidebar:
    st.divider()
    st.markdown("### 🏀 Team Selector")
    selected_team_name = st.selectbox("Select Team", ["None"] + sorted(team_options.keys()))
    
    if selected_team_name != "None":
        tid = team_options[selected_team_name]
        st.info(f"Fetching {selected_team_name} Roster...")
        roster_df = get_roster(tid)
        
        if not roster_df.empty:
            st.markdown("### Active Roster")
            st.markdown("*(Click player to analyze)*")
            for _, row in roster_df.iterrows():
                if st.button(row['PLAYER'], key=f"btn_{row['PLAYER_ID']}"):
                    st.session_state.selected_player_id = row['PLAYER_ID']
                    st.session_state.selected_player_name = row['PLAYER']
        else:
            st.error("Could not fetch roster.")

# --- MAIN PAGE LOGIC ---
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🏀 NBA PROP ASSASSIN</h1>", unsafe_allow_html=True)

if 'selected_player_id' not in st.session_state:
    st.session_state.selected_player_id = None
    st.session_state.selected_player_name = None

# Manual Search Fallback
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    manual_search = st.text_input("", placeholder="Or search manually (e.g. Luka Doncic)...", label_visibility="collapsed")
    if st.button("Search", use_container_width=True) and manual_search:
        active_players = players.get_active_players()
        found = next((p for p in active_players if p['full_name'].lower() == manual_search.lower()), None)
        if found:
            st.session_state.selected_player_id = found['id']
            st.session_state.selected_player_name = found['full_name']
        else:
            st.error("Player not found.")

# Display Analysis
if st.session_state.selected_player_id:
    pid = st.session_state.selected_player_id
    pname = st.session_state.selected_player_name
    
    recent_df = get_player_recent_stats(pid)
    
    if not recent_df.empty and len(recent_df) >= 5:
        preds = predict(models, recent_df)
        
        # --- CALCULATE LAST 5 AVERAGES ---
        l5 = recent_df.tail(5)
        avg_pts = l5['PTS'].mean()
        avg_reb = l5['REB'].mean()
        avg_ast = l5['AST'].mean()
        avg_stl = l5['STL'].mean()
        avg_blk = l5['BLK'].mean()
        
        # --- PLAYER HEADER WITH STATS ---
        with st.container():
            st.markdown(f"""
            <div class='glass-card' style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;'>
                <div style='display: flex; align-items: center; gap: 20px;'>
                    <img src='https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png' style='border-radius: 10px; height: 100px;'>
                    <div>
                        <div class='player-name'>{pname}</div>
                        <div class='team-name'>ID: {pid} • 2025-26 Season</div>
                    </div>
                </div>
                
                <div style='display: flex; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 10px;'>
                    <div class='banner-stat-box'>
                        <div class='banner-val'>{avg_pts:.1f}</div>
                        <div class='banner-label'>PTS</div>
                    </div>
                    <div class='banner-stat-box'>
                        <div class='banner-val'>{avg_reb:.1f}</div>
                        <div class='banner-label'>REB</div>
                    </div>
                    <div class='banner-stat-box'>
                        <div class='banner-val'>{avg_ast:.1f}</div>
                        <div class='banner-label'>AST</div>
                    </div>
                    <div class='banner-stat-box'>
                        <div class='banner-val'>{avg_stl:.1f}</div>
                        <div class='banner-label'>STL</div>
                    </div>
                    <div class='banner-stat-box'>
                        <div class='banner-val'>{avg_blk:.1f}</div>
                        <div class='banner-label'>BLK</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Betting Analyzer
        st.markdown("### 📊 Prop Analyzer")
        tab_pts, tab_reb, tab_ast = st.tabs(["Points", "Rebounds", "Assists"])
        
        def render_card(stat_name, pred_val, df, stat_col):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"<div class='glass-card'><div class='stat-label'>AI Pred</div><div class='big-stat'>{pred_val}</div></div>", unsafe_allow_html=True)
                line = st.number_input(f"{stat_name} Line", value=float(int(pred_val)), step=0.5, key=f"line_{stat_name}_{pid}")
                diff = pred_val - line
                if diff > 0: st.markdown(f"<span class='badge-over'>BET OVER</span> (+{diff:.1f})", unsafe_allow_html=True)
                elif diff < 0: st.markdown(f"<span class='badge-under'>BET UNDER</span> ({diff:.1f})", unsafe_allow_html=True)
                
            with c2:
                fig = go.Figure()
                last_10 = df.tail(10)
                fig.add_trace(go.Scatter(x=last_10['GAME_DATE'], y=last_10[stat_col], fill='tozeroy', mode='lines+markers', line=dict(color='#00ADB5', width=3), name='Actual'))
                fig.add_hline(y=line, line_dash="dash", line_color="white", annotation_text="Line")
                fig.add_hline(y=pred_val, line_dash="dot", line_color="#00ADB5", annotation_text="AI")
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=30,l=10,r=10,b=10))
                st.plotly_chart(fig, use_container_width=True)

        with tab_pts: render_card("Points", preds['PTS'], recent_df, 'PTS')
        with tab_reb: render_card("Rebounds", preds['REB'], recent_df, 'REB')
        with tab_ast: render_card("Assists", preds['AST'], recent_df, 'AST')

    else:
        st.warning("Player inactive or insufficient data (needs 5 games).")
