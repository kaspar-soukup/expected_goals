import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from mplsoccer import Pitch
import matplotlib.pyplot as plt

st.set_page_config(page_title="Expected Goals", layout="wide")
st.title("⚽ Expected Goals Predictor")

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(Path(__file__).parent / "data/processed/catboost_model_custom.cbm")
    return model

model = load_model()

if 'player_y' not in st.session_state:
    st.session_state.player_y = 100.0
    st.session_state.player_x = 40.0
    st.session_state.goalkeeper_y = 120.0
    st.session_state.goalkeeper_x = 40.0

col_pitch, col_inputs = st.columns([1.5, 1])

with col_pitch:
    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_type='statsbomb', line_color='black', pitch_color='white')
    pitch.draw(ax=ax)
    
    ax.scatter(st.session_state.player_y, st.session_state.player_x, 
               s=400, c='yellow', edgecolors='black', linewidth=2, label='Shooter')
    ax.scatter(st.session_state.goalkeeper_y, st.session_state.goalkeeper_x, 
               s=400, c='red', edgecolors='black', linewidth=2, label='Goalkeeper')
    
    ax.legend(loc='upper right')
    ax.set_title("Pitch View")
    st.pyplot(fig, use_container_width=True)

with col_inputs:
    st.subheader("Shot Details")
    
    st.session_state.player_y = st.slider("Shooter Distance", 60.0, 120.0, st.session_state.player_y)
    st.session_state.player_x = st.slider("Shooter Width", 0.0, 80.0, st.session_state.player_x)
    st.session_state.goalkeeper_y = st.slider("GK Distance", 100.0, 120.0, st.session_state.goalkeeper_y)
    st.session_state.goalkeeper_x = st.slider("GK Width", 0.0, 80.0, st.session_state.goalkeeper_x)
    
    st.write("**Conditions:**")
    under_pressure = st.checkbox("Under Pressure")
    open_goal = st.checkbox("Open Goal")
    first_time = st.checkbox("First Time")
    one_on_one = st.checkbox("One on One")
    
    st.write("**Type:**")
    body_part = st.radio("Body Part", ["foot", "head", "other"])
    shot_technique = st.radio("Technique", ["normal", "backheel", "diving_header", "half_volley", "lob", "overhead_kick", "volley"])
    shot_type = st.radio("Shot Type", ["open_play", "free_kick"])
    play_pattern = st.radio("Pattern", ["regular_play", "from_corner", "from_counter", "from_free_kick", "from_goal_kick", "from_keeper", "from_kick_off", "from_throw_in", "other"])

if st.button("Predict", type="primary", use_container_width=True):
    player_y = st.session_state.player_y
    player_x = st.session_state.player_x
    gk_y = st.session_state.goalkeeper_y
    gk_x = st.session_state.goalkeeper_x
    
    features = {
        'under_pressure': int(under_pressure),
        'shot_open_goal': int(open_goal),
        'shot_first_time': int(first_time),
        'shot_one_on_one': int(one_on_one),
        'player_y': player_y,
        'player_x': player_x,
        'goalkeeper_y': gk_y,
        'goalkeeper_x': gk_x,
        'distance_from_goal_center': np.sqrt((120 - player_y) ** 2 + (40 - player_x) ** 2),
        'gk_distance_from_goal_center': np.sqrt((120 - gk_y) ** 2 + (40 - gk_x) ** 2),
        'distance_player_gk': np.sqrt((player_y - gk_y) ** 2 + (player_x - gk_x) ** 2),
        'shot_angle': np.arctan2(abs(player_x - 40), abs(120 - player_y)),
        'body_part_foot': int(body_part == "foot"),
        'body_part_head': int(body_part == "head"),
        'body_part_other': int(body_part == "other"),
        'shot_technique_backheel': int(shot_technique == "backheel"),
        'shot_technique_diving_header': int(shot_technique == "diving_header"),
        'shot_technique_half_volley': int(shot_technique == "half_volley"),
        'shot_technique_lob': int(shot_technique == "lob"),
        'shot_technique_normal': int(shot_technique == "normal"),
        'shot_technique_overhead_kick': int(shot_technique == "overhead_kick"),
        'shot_technique_volley': int(shot_technique == "volley"),
        'shot_type_free_kick': int(shot_type == "free_kick"),
        'shot_type_open_play': int(shot_type == "open_play"),
        'play_pattern_from_corner': int(play_pattern == "from_corner"),
        'play_pattern_from_counter': int(play_pattern == "from_counter"),
        'play_pattern_from_free_kick': int(play_pattern == "from_free_kick"),
        'play_pattern_from_goal_kick': int(play_pattern == "from_goal_kick"),
        'play_pattern_from_keeper': int(play_pattern == "from_keeper"),
        'play_pattern_from_kick_off': int(play_pattern == "from_kick_off"),
        'play_pattern_from_throw_in': int(play_pattern == "from_throw_in"),
        'play_pattern_regular_play': int(play_pattern == "regular_play"),
        'play_pattern_other': int(play_pattern == "other"),
    }
    
    X = pd.DataFrame([features])
    pred = model.predict_proba(X)[0][1]
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Goal Probability", f"{pred:.1%}")
    col2.metric("Distance to Goal", f"{features['distance_from_goal_center']:.1f}m")
    col3.metric("xG", f"{pred:.3f}")
