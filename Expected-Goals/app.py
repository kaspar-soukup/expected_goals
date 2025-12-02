import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
except ImportError:
    st.error("❌ CatBoost not installed. Please run: pip install catboost")
    st.stop()

# Set page config
st.set_page_config(page_title="Expected Goals Predictor", layout="wide")

st.title("⚽ Expected Goals Predictor")
st.write("Predict the probability of a shot resulting in a goal")

# Load the trained model and preprocessor
@st.cache_resource
def load_model_and_scaler():
    PROJECT_ROOT = Path(__file__).parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    
    # Load the CatBoost model
    model = CatBoostClassifier()
    model.load_model(PROCESSED_DIR / "catboost_model_custom.cbm")
    
    # Load the processed data to get the scaler
    football_model_df = pd.read_pickle(PROCESSED_DIR / "football_model_processed.pickle")
    
    # Get feature columns (same as training)
    X = football_model_df.drop(columns={'shot_outcome_encoded', 'body_part_other', 'shot_technique_lob', 
                                         'play_pattern_other', 'shot_type_free_kick', 
                                         'distance_from_goal_left_post', 'distance_from_goal_right_post',
                                         'player_x', 'goalkeeper_x', 'gk_distance_from_goal_left_post',
                                         'gk_distance_from_goal_right_post'})
    
    # Identify continuous columns for scaling
    cont_cols = [col for col in X.columns 
                 if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 2]
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X[cont_cols])
    
    return model, scaler, X.columns.tolist(), cont_cols

try:
    model, scaler, feature_columns, cont_cols = load_model_and_scaler()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model_loaded = False

if model_loaded:
    from mplsoccer import Pitch
    import matplotlib.pyplot as plt
    
    # Initialize session state for positions
    if 'player_y' not in st.session_state:
        st.session_state.player_y = 100.0
        st.session_state.player_x = 40.0
        st.session_state.goalkeeper_y = 120.0
        st.session_state.goalkeeper_x = 40.0
    
    # Display football pitch with positions
    st.subheader("⚽ Pitch & Shot Details")
    
    col_pitch, col_inputs = st.columns([1.5, 1])
    
    with col_pitch:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create pitch
        pitch = Pitch(pitch_type='statsbomb', line_color='black', line_zorder=2, pitch_color='white')
        pitch.draw(ax=ax)
        
        # Plot positions as scatter
        ax.scatter(st.session_state.player_y, st.session_state.player_x, 
                  s=400, c='yellow', edgecolors='black', linewidth=2, 
                  label='Shooter (Yellow)', marker='o', zorder=5)
        ax.scatter(st.session_state.goalkeeper_y, st.session_state.goalkeeper_x, 
                  s=400, c='red', edgecolors='black', linewidth=2, 
                  label='Goalkeeper (Red)', marker='s', zorder=5)
        
        ax.legend(loc='upper right', fontsize=11)
        ax.set_title("Pitch View (0-120 x 0-80)", fontsize=12)
        
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 Use the inputs below to position players:")
        
        col_pos1, col_pos2 = st.columns(2)
        with col_pos1:
            st.session_state.player_y = st.number_input("Shooter Distance (0-120)", min_value=0.0, max_value=120.0, value=st.session_state.player_y, step=1.0, key="shooter_dist")
            st.session_state.goalkeeper_y = st.number_input("GK Distance (0-120)", min_value=0.0, max_value=120.0, value=st.session_state.goalkeeper_y, step=1.0, key="gk_dist")
        
        with col_pos2:
            st.session_state.player_x = st.number_input("Shooter Width (0-80)", min_value=0.0, max_value=80.0, value=st.session_state.player_x, step=1.0, key="shooter_width")
            st.session_state.goalkeeper_x = st.number_input("GK Width (0-80)", min_value=0.0, max_value=80.0, value=st.session_state.goalkeeper_x, step=1.0, key="gk_width")
    
    with col_inputs:
        st.subheader("🎯 Shot Details")
        
        # Boolean features
        st.write("**Conditions:**")
        under_pressure = st.checkbox("Under Pressure")
        shot_open_goal = st.checkbox("Open Goal")
        shot_first_time = st.checkbox("First Time Shot")
        shot_one_on_one = st.checkbox("One on One")
        
        st.divider()
        
        # Categorical features
        st.write("**Characteristics:**")
        body_part = st.radio("Body Part", ["foot", "head"])
        
        shot_technique = st.selectbox(
            "Shot Technique",
            ["normal", "backheel", "diving_header", "half_volley", "overhead_kick", "volley"]
        )
        
        st.divider()
        
        st.write("**Context:**")
        shot_type = st.radio("Shot Type", ["open_play"])
        
        play_pattern = st.selectbox(
            "Play Pattern",
            ["regular_play", "from_corner", "from_counter", "from_free_kick", 
             "from_goal_kick", "from_keeper", "from_kick_off", "from_throw_in"]
        )
    
    st.divider()
    
    player_y = st.session_state.player_y
    player_x = st.session_state.player_x
    goalkeeper_y = st.session_state.goalkeeper_y
    goalkeeper_x = st.session_state.goalkeeper_x

    # Create prediction button
    if st.button("🔮 Predict Goal Probability", type="primary", use_container_width=True):
        # Build input dataframe with all required features
        input_data = {}
        
        # Add binary features
        input_data['under_pressure'] = int(under_pressure)
        input_data['shot_open_goal'] = int(shot_open_goal)
        input_data['shot_first_time'] = int(shot_first_time)
        input_data['shot_one_on_one'] = int(shot_one_on_one)
        
        # Add position features
        input_data['player_y'] = player_y
        input_data['goalkeeper_y'] = goalkeeper_y
        
        # Calculate distances
        input_data['distance_from_goal_center'] = np.sqrt((120 - player_y) ** 2 + (40 - player_x) ** 2)
        input_data['gk_distance_from_goal_center'] = np.sqrt((120 - goalkeeper_y) ** 2 + (40 - goalkeeper_x) ** 2)
        input_data['distance_player_gk'] = np.sqrt((player_y - goalkeeper_y) ** 2 + (player_x - goalkeeper_x) ** 2)
        
        # Calculate shot angle (simplified: angle to goal center)
        input_data['shot_angle'] = np.arctan2(abs(player_x - 40), abs(120 - player_y))
        
        # Add body part dummies
        input_data['body_part_foot'] = int(body_part == "foot")
        input_data['body_part_head'] = int(body_part == "head")
        
        # Add shot technique dummies
        input_data['shot_technique_backheel'] = int(shot_technique == "backheel")
        input_data['shot_technique_diving_header'] = int(shot_technique == "diving_header")
        input_data['shot_technique_half_volley'] = int(shot_technique == "half_volley")
        input_data['shot_technique_normal'] = int(shot_technique == "normal")
        input_data['shot_technique_overhead_kick'] = int(shot_technique == "overhead_kick")
        input_data['shot_technique_volley'] = int(shot_technique == "volley")
        
        # Add shot type dummies
        input_data['shot_type_open_play'] = int(shot_type == "open_play")
        
        # Add play pattern dummies
        input_data['play_pattern_from_corner'] = int(play_pattern == "from_corner")
        input_data['play_pattern_from_counter'] = int(play_pattern == "from_counter")
        input_data['play_pattern_from_free_kick'] = int(play_pattern == "from_free_kick")
        input_data['play_pattern_from_goal_kick'] = int(play_pattern == "from_goal_kick")
        input_data['play_pattern_from_keeper'] = int(play_pattern == "from_keeper")
        input_data['play_pattern_from_kick_off'] = int(play_pattern == "from_kick_off")
        input_data['play_pattern_from_throw_in'] = int(play_pattern == "from_throw_in")
        input_data['play_pattern_regular_play'] = int(play_pattern == "regular_play")
        
        # Create dataframe with all features in correct order
        X_input = pd.DataFrame([input_data])
        
        # Ensure all expected features are present (set missing ones to 0)
        for col in feature_columns:
            if col not in X_input.columns:
                X_input[col] = 0
        
        # Select only the features in the correct order
        X_input = X_input[feature_columns]
        
        # Scale continuous features
        X_input_scaled = X_input.copy()
        X_input_scaled[cont_cols] = scaler.transform(X_input[cont_cols])
        
        # Make prediction
        prediction = model.predict_proba(X_input_scaled)[0][1]
        
        # Display result
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Goal Probability", f"{prediction:.1%}")
        
        with col2:
            if prediction > 0.5:
                assessment = "🟢 High"
                description = "Likely Goal"
            elif prediction > 0.15:
                assessment = "🟡 Medium"
                description = "Possible"
            else:
                assessment = "🔴 Low"
                description = "Unlikely"
            st.metric("Assessment", assessment, description)
        
        with col3:
            distance = np.sqrt((120 - player_y) ** 2 + (40 - player_x) ** 2)
            st.metric("Distance to Goal", f"{distance:.1f} meters")
