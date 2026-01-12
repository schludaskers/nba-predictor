# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2, playergamelog
from nba_api.stats.static import players

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NBA Prop Assassin", layout="wide", page_icon="🏀")

# Custom CSS for Dark/Neon Theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #41444b;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .big-number {
        font-size: 32px;
        font-weight: bold;
        color: #00ADB5;
    }
    .stat-label {
        font-size: 14px;
        color: #AAAAAA;
    }
</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_models():
    return joblib.load('nba_prop_models.pkl')


def get_todays_games(date_str):
    # Fetch games for the selected date
    board = scoreboardv2.ScoreboardV2(game_date=date_str)
    games = board.game_header.get_data_frame()
    return games


def get_player_recent_stats(player_id):
    # Fetch current season stats to calculate recent form
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
        df = gamelog.get_data_frames()[0]

        if df.empty:
            return pd.DataFrame()

        # FIX: Normalize columns to uppercase
        df.columns = df.columns.str.upper()

        # Sort by date ascending (oldest to newest)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
        df = df.sort_values(by='GAME_DATE')
        return df
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return pd.DataFrame()


def predict_props(models, recent_stats):
    if len(recent_stats) < 5:
        return None  # Not enough data

    # Calculate features (Last 5 average) just like we did in training
    last_5 = recent_stats.tail(5)
    features = {
        'PTS_L5': [last_5['PTS'].mean()],
        'REB_L5': [last_5['REB'].mean()],
        'AST_L5': [last_5['AST'].mean()],
        'MIN_L5': [last_5['MIN'].mean()]
    }
    feature_df = pd.DataFrame(features)

    predictions = {}
    for stat, model in models.items():
        pred = model.predict(feature_df)[0]
        predictions[stat] = round(pred, 1)

    return predictions


# --- MAIN APP UI ---

st.title("🏀 NBA Prop Predictor")
st.markdown("### AI-Powered Predictions for Today's Slate")

# Sidebar for controls
st.sidebar.header("Settings")
selected_date = st.sidebar.date_input("Select Date", datetime.now())

# Load Models
try:
    models = load_models()
except FileNotFoundError:
    st.error("Model file not found! Please run 'train_model.py' first.")
    st.stop()

# Get Games
with st.spinner(f"Fetching games for {selected_date}..."):
    games_df = get_todays_games(selected_date)

if games_df.empty:
    st.warning("No games found for this date.")
else:
    # Display Games List
    st.sidebar.markdown("### Games Today")
    game_list = [f"{row['HOME_TEAM_ID']} vs {row['VISITOR_TEAM_ID']}" for i, row in games_df.iterrows()]
    # Note: Using Team IDs here is simple; in a real app you'd map IDs to Names (Lakers, Celtics, etc.)
    st.sidebar.write(f"Found {len(games_df)} games.")

    # Main Area
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔎 Player Lookup")
        player_name = st.text_input("Enter Player Name", placeholder="e.g. LeBron James")

        predict_btn = st.button("Generate Prediction", type="primary")

        if predict_btn and player_name:
            # Find player ID
            active_players = players.get_active_players()
            player_info = next((p for p in active_players if p['full_name'].lower() == player_name.lower()), None)

            if player_info:
                st.success(f"Found: {player_info['full_name']}")

                with st.spinner("Analyzing last 5 games..."):
                    recent_df = get_player_recent_stats(player_info['id'])

                    if recent_df is not None and not recent_df.empty:
                        preds = predict_props(models, recent_df)

                        if preds:
                            # --- RESULTS DISPLAY ---
                            st.markdown(f"## 🔮 Predictions for {player_info['full_name']}")

                            c1, c2, c3 = st.columns(3)
                            c1.markdown(
                                f"<div class='metric-card'><div class='big-number'>{preds['PTS']}</div><div class='stat-label'>PTS Prediction</div></div>",
                                unsafe_allow_html=True)
                            c2.markdown(
                                f"<div class='metric-card'><div class='big-number'>{preds['REB']}</div><div class='stat-label'>REB Prediction</div></div>",
                                unsafe_allow_html=True)
                            c3.markdown(
                                f"<div class='metric-card'><div class='big-number'>{preds['AST']}</div><div class='stat-label'>AST Prediction</div></div>",
                                unsafe_allow_html=True)

                            # --- TREND GRAPH ---
                            st.markdown("### Recent Form (Last 5 Games)")

                            last_5 = recent_df.tail(5)
                            fig = go.Figure()

                            # Points Line
                            fig.add_trace(go.Scatter(
                                x=last_5['GAME_DATE'],
                                y=last_5['PTS'],
                                mode='lines+markers',
                                name='Actual PTS',
                                line=dict(color='#00ADB5', width=3)
                            ))

                            # Prediction Line (dashed)
                            fig.add_hline(y=preds['PTS'], line_dash="dash", line_color="#FF2E63",
                                          annotation_text="Projected")

                            fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=350,
                                margin=dict(l=20, r=20, t=30, b=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        else:
                            st.error("Not enough recent games (need at least 5) to make a prediction.")
                    else:
                        st.error("Could not fetch stats. Check internet connection or if player is active this season.")
            else:
                st.error("Player not found. Check spelling.")

    with col2:
        st.info(
            "💡 **How it works:** This model looks at the player's last 5 games (Minutes, PTS, REB, AST) and uses a Random Forest Regressor trained on 2024-2026 data to forecast tonight's stat line.")