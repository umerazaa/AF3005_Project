import streamlit as st 
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score, mean_squared_error
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Hall of Justice", layout="wide")

# --- Your chosen backgrounds for each hero ---
HERO_BACKGROUNDS = {
    "Batman": "https://wallpapercave.com/wp/wp14140690.jpg",
    "Superman": "https://images.hdqwalls.com/download/superman-fortress-of-solitude-to-3840x2160.jpg",
    "Flash": "https://wallpapercave.com/wp/wp10423270.jpg",
    "Green Lantern": "https://wallpapercave.com/wp/wp7719013.jpg",
    "Default": "https://wallpapersok.com/images/high/glowing-symbol-justice-league-6dqduybwd0de96tg.webp"
}

# --- Hero Configuration ---
HEROES = {
     "Batman": {
        "id": "0001",
        "welcome": "RECOGNIZED. JL MEMBER 0001. WELCOME BATMAN!",
        "location": "Batcave",
        "passage_lines": [
            "Welcome to the Batcave.",
            "Time to analyze Financial Data in the shadows using the Batcomputer.",
            "I'm Batman!",
            "Your guide through Machine Learning.",
            "Together, we’ll hunt down bugs like rogues in Arkham...",
            "Turning lines of code into the Bat-Tools you need for sophisticated financial analysis."
        ],
        "farewell_lines": ["Stay vigilant. Batman out."]
    },
    "Superman": {
        "id": "0002",
        "welcome": "RECOGNIZED. JL MEMBER 0002. WELCOME SUPERMAN!",
        "location": "Fortress of Solitude",
        "passage_lines": [
            "Need some help stranger?",
            "From the Fortress of Solitude to Metropolis, we'll soar through regression.",
            "I'm Superman.",
            "Ready to lift heavy datasets, bringing order to your models faster than a speeding bullet!"
        ],
        "farewell_lines": ["Hope is always alive. Until next time!"]
    },
    "Flash": {
        "id": "0003",
        "welcome": "RECOGNIZED. JL MEMBER 0003. WELCOME THE FLASH!",
        "location": "Speed Force",
        "passage_lines": [
            "Welcome to my speed lab!",
            "Ready to move at the speed of light?",
            "I'm The Flash.",
            "Let’s zoom through data and deliver lightning-fast insights, processing market data in a flash."
        ],
        "farewell_lines": ["Gotta run!"]
    },
    "Green Lantern": {
        "id": "0004",
        "welcome": "RECOGNIZED. JL MEMBER 0004. WELCOME GREEN LANTERN!",
        "location": "Green Lantern Corps",
        "passage_lines": [
            "In brightest day, in blackest night...",
            "Your willpower is your superpower.",
            "I'm Green Lantern.",
            "Let’s construct code constructs, powering through clustering and classification with determination."
        ],
        "farewell_lines": ["In brightest day or blackest night, see you next time."]
    }
}

# Templates for hero interpretation lines
INTERPRET_TEMPLATES = {
    "Batman": {
         "Linear Regression": "With an R² of {metric:.4f}, our predictions hit their mark. Gotham is safer tonight.",
        "Logistic Regression": "Accuracy at {metric:.4f}. Another victory for the Justice League.",
        "K-Means Clustering": "Clusters found: {k}. We'll keep watch on each sector."
    },
    "Superman": {
         "Linear Regression": "An R² of {metric:.4f}—truth and justice are clearly on our side.",
        "Logistic Regression": "Accuracy at {metric:.4f}. We'll keep Metropolis safe.",
        "K-Means Clustering": "Great work! {k} groups identified—strength in numbers."
    },
    "Flash": {
        "Linear Regression": "Zoom! R² hit {metric:.4f}. We're lightning fast!",
        "Logistic Regression": "Accuracy at {metric:.4f}. Speed isn't our only power.",
        "K-Means Clustering": "We mapped {k} clusters in a flash!"
    },
    "Green Lantern": {
          "Linear Regression": "R² of {metric:.4f}. My will constructs victory from data.",
        "Logistic Regression": "Accuracy at {metric:.4f}. The Corps would be proud.",
        "K-Means Clustering": "By willpower, {k} clusters reveal the data's structure."
    }
}

def get_bg_url():
    hero = st.session_state.get("hero_choice", None)
    if hero and hero in HERO_BACKGROUNDS:
        return HERO_BACKGROUNDS[hero]
    else:
        return HERO_BACKGROUNDS["Default"]

# --- Load Google Fonts and Comic CSS with dynamic background ---
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@700;900&display=swap" rel="stylesheet">
<style>
    body, .stApp {{
        background-color: #18172a;
        background-image: url('{get_bg_url()}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #FFD700;
    }}
    .comic-panel {{
        border: 8px solid #FFD700;
        border-radius: 32px;
        box-shadow: 0 0 40px #18172a, 0 0 12px #FFD700;
        margin: 2.2rem auto 2.5rem auto;
        padding: 2.2rem 2.8rem 2.6rem 2.8rem;
        max-width: 920px;
        background: rgba(20,20,30,0.80);
        position: relative;
        text-align: center;
    }}
    .gotham-type {{
        font-family: 'Oswald', Impact, Arial Black, Arial, sans-serif;
        font-size: 4rem;
        letter-spacing: 2px;
        color: #FFD700;
        font-weight: 900;
        text-shadow: 0 0 30px #FFD700, 0 0 10px #18172a, 2px 2px 0 #18172a, 4px 4px 12px #000000;
        text-transform: uppercase;
        line-height: 1.1;
        margin-bottom: 1rem;
        margin-top: 0;
    }}
    .speech-bubble {{
        background: #18172a;
        border-radius: 23px;
        display: inline-block;
        padding: 1.1rem 2.5rem;
        box-shadow: 0 0 18px #FFD70066;
        margin-top: 1.7rem;
        margin-bottom: 0.9rem;
        color: #ffe066;
        font-size: 1.38rem;
        font-family: 'Oswald', Arial, sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        position: relative;
    }}
    .speech-bubble:after {{
        content: '';
        position: absolute;
        left: 65px;
        bottom: -26px;
        width: 0; height: 0;
        border: 16px solid transparent;
        border-top: 18px solid #18172a;
        filter: drop-shadow(0 0 6px #FFD70088);
    }}
    .comic-issue-badge {{
        position: absolute;
        top: -26px;
        left: -26px;
        z-index: 9;
        background: #00FFD5;
        color: #222;
        font-family: 'Oswald', Arial Black, Arial, sans-serif;
        font-size: 1.45rem;
        padding: 0.23rem 1.25rem 0.15rem 1.2rem;
        border-radius: 18px 8px 24px 8px;
        box-shadow: 0 2px 8px #18172a99;
        border: 3.5px solid #FFD700;
        letter-spacing: 1px;
        font-weight: 900;
    }}
    .comic-btn button {{
        background: #FFD700;
        color: #18172a;
        font-family: 'Oswald', Arial, sans-serif;
        font-weight: 900;
        font-size: 1.35rem;
        border-radius: 11px;
        padding: 1.1rem 2.1rem;
        border: none;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        transition: all .2s cubic-bezier(.4,2,.45,1.4);
        box-shadow: 0 0 0px #FFD700, 0 0 12px #FFD70099;
        text-transform: uppercase;
    }}
    .comic-btn button:hover {{
        background: #ffe066;
        color: #18172a;
        transform: scale(1.08) rotate(-2.5deg);
        box-shadow: 0 0 20px #FFD70088;
        cursor: pointer;
    }}
     h1, h2, h3, .stApp h1 {{
        color: #FFD700 !important;
        font-family: 'Oswald', Arial Black, Arial, sans-serif;
        font-weight: 900;
        text-shadow: 0 0 18px #FFD700;
        text-transform: uppercase;
    }}
    .upload-icon {{
        width: 120px;
        height: 120px;
        margin: 0.5rem auto;
        display: block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 10px #FFD700);
    }}
    @keyframes float {{
        0%, 100% {{transform: translateY(0);}}
        50% {{transform: translateY(-15px);}}
    }}
</style>
""", unsafe_allow_html=True)

# --- Typewriter Utility Functions ---
def typewriter(lines, color="#FFD700", delay=0.028):
    area = st.empty()
    buffer = ""
    for line in lines:
        for c in line:
            buffer += c
            area.markdown(
                f"<div class='speech-bubble' style='background:#18172a; color:{color}; font-family:Oswald,Arial,sans-serif; font-size:1.5rem; font-weight:800; letter-spacing:1.2px;'>{buffer}</div>",
                unsafe_allow_html=True,
            )
            time.sleep(delay)
        buffer += "\n"


def typewriter_welcome(text, color="#FFD700", delay=0.065):
    area = st.empty()
    buffer = ""
    for c in text:
        buffer += c
        area.markdown(
            f"<div class='gotham-type' style='font-size:2.4rem; margin-bottom:1.2rem; color:{color}; text-shadow:0 0 30px {color}, 0 0 10px #18172a, 2px 2px 0 #18172a, 4px 4px 12px #000000;'>{buffer}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(delay)


def typewriter_intro_panel(text, delay=0.065):
    panel = st.empty()
    buffer = ""
    for c in text:
        buffer += c
        panel.markdown(
            f"""
            <div class="comic-panel">
                <div class="comic-issue-badge">ISSUE #1</div>
                <div class="gotham-type">{buffer}</div>
                <div class="speech-bubble">
                    The command center of Earth's greatest defenders. Step forward, hero — your mission awaits.
                </div>
                <div class="hero-row">
                    <img src="https://img.icons8.com/color/96/batman--v1.png" class="hero-icon" />
                    <img src="https://img.icons8.com/color/96/superman.png" class="hero-icon" />
                    <img src="https://img.icons8.com/color/96/green-lantern.png" class="hero-icon" />
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(delay)


# --- Session State for Navigation ---
if 'step' not in st.session_state:
    st.session_state.step = 'intro'

if st.session_state.step == 'intro':
    typewriter_intro_panel("WELCOME TO THE HALL OF JUSTICE")

    # Center the Streamlit button with flexbox and style it
    st.markdown("""
     <style>
    .center-btn {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-top: -1.5rem;
        margin-bottom: 2.7rem;
    }
    .hero-row {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin-top: 1rem;
    }
    .hero-icon {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        border: 4px solid #FFD700;
        box-shadow: 0 0 10px #FFD700;
        animation: bounce 1.8s infinite;
    }
    @keyframes bounce {
        0%, 20%, 60%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-25%);
        }
        80% {
            transform: translateY(-10%);
        }
    }
    .hero-row {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin-top: 1rem;
    }
    .hero-icon {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        border: 4px solid #FFD700;
        box-shadow: 0 0 10px #FFD700;
        animation: hero-bounce 3s ease-in-out infinite;
    }
    @keyframes hero-bounce {
        0%, 100% {transform: translateY(0);}
        50% {transform: translateY(-12px);}
    }
    .hero-row {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin-top: 1rem;
    }
    .hero-icon {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        border: 4px solid #FFD700;
        box-shadow: 0 0 10px #FFD700;
    }
    .stButton > button {
        background: #FFD700;
        color: #18172a;
        font-family: 'Oswald', Arial, sans-serif;
        font-size: 2rem;
        font-weight: 900;
        padding: 1.2rem 3.5rem;
        border: none;
        border-radius: 14px;
        box-shadow: 0 0 38px 12px #FFD700BB, 0 0 6px #222;
        letter-spacing: 2px;
        text-transform: uppercase;
        cursor: pointer;
        transition: 0.18s all cubic-bezier(.4,2,.45,1.4);
    }
    .stButton > button:hover {
        background: #ffe066 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Center the button using a container and custom class
    st.markdown('<div class="center-btn">', unsafe_allow_html=True)
    if st.button("Enter the Hall", key="enter_btn"):
        st.session_state.step = 'upload'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)



# --- Step 1: Dataset Upload & Hero Selection ---
elif st.session_state.step == 'upload':
    st.markdown(f"""
    <div class="comic-panel" style="max-width:700px;">
        <div class="comic-issue-badge">STEP 1</div>
        <div class="gotham-type" style="font-size:2.6rem; margin-bottom:0.4rem;">Upload Your Dataset</div>
        <img src="https://img.icons8.com/color/120/csv.png" class="upload-icon" />
        <div class="speech-bubble">Bring in a CSV file to get the adventure started!</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV Dataset:", type=["csv"])
    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.step = 'select_hero'
        st.rerun()

# --- Step 2: Hero Selection ---
elif st.session_state.step == 'select_hero':
    st.markdown(f"""
    <div class="comic-panel" style="max-width:700px;">
        <div class="comic-issue-badge">CHOOSE HERO</div>
        <div class="gotham-type" style="font-size:2.6rem; margin-bottom:0.4rem;">Step 2: Which Superhero's Help Do You Need?</div>
    </div>
    """, unsafe_allow_html=True)
    hero_choice = st.selectbox("Choose your JL Member:", list(HEROES.keys()))
    if st.button("Deploy Hero"):
        st.session_state.hero_choice = hero_choice
        st.session_state.step = 'ml_task'
        st.rerun()

# --- Step 3: ML Task Selection & Workflow ---
elif st.session_state.step == 'ml_task':
    hero_choice = st.session_state.hero_choice
    hero = HEROES[hero_choice]

    # --- Dynamic Comic Colors ---
    HERO_COLORS = {
        "Batman": {"primary": "#FFD700", "secondary": "#FFD700", "accent": "#22223B"},
        "Superman": {"primary": "#54f6ff", "secondary": "#54f6ff", "accent": "#54f6ff"},
        "Flash": {"primary": "#FF4500", "secondary": "#FF4500", "accent": "#FFE066"},
        "Green Lantern": {"primary": "#00FF41", "secondary": "#222", "accent": "#8BFF95"}
    }

    hero_color = HERO_COLORS.get(hero_choice, HERO_COLORS["Batman"])

    # --- Go Back Button ---
    if st.button("← Go Back to Hero Select", key="back_to_hero"):
        st.session_state.step = 'select_hero'
        # So you see the intro on new hero, but not when going back
        st.session_state['intro_played'] = False
        st.rerun()

    
    # Use global typewriter helpers

    if 'intro_played' not in st.session_state or st.session_state.get('prev_hero') != hero_choice:
        st.session_state['intro_played'] = True
        st.session_state['prev_hero'] = hero_choice
        typewriter_welcome(hero['welcome'], color=hero_color['primary'])
        typewriter(hero['passage_lines'], color=hero_color['primary'])
    else:
        typewriter_welcome(hero['welcome'], color=hero_color['primary'])
        for line in hero['passage_lines']:
            st.markdown(
                f"<div class='speech-bubble' style='background:#18172a; color:{hero_color['primary']}; font-family:Oswald,Arial,sans-serif; font-size:1.5rem; font-weight:800; letter-spacing:1.2px;'>{line}</div>",
                unsafe_allow_html=True
            )

    st.sidebar.markdown(
        f"<div style='font-family:Oswald,Arial,sans-serif; font-size:2rem; color:#FFD700; letter-spacing:1.2px; font-weight:900; text-shadow:0 0 15px #FFD700;'>Machine Learning Tasks</div>",
        unsafe_allow_html=True
    )
    task = st.sidebar.radio(
        "Which task do you want to perform?",
        ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
    )

    df = st.session_state.data
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("### Data Preview")
    st.dataframe(df.head())

    if task in ["Linear Regression", "Logistic Regression"]:
        st.subheader(f"{task} Setup")
        target_col = st.selectbox("Select Target Variable:", numeric_cols, key="target")
        feature_cols = st.multiselect(
            "Select Feature(s):", [col for col in numeric_cols if col != target_col], key="features"
        )
        test_size = st.slider("Test set size (fraction)", 0.1, 0.5, 0.2)
        if st.button(f"Train {task}"):
            if not feature_cols or not target_col:
                st.error("Please select at least one feature and a target.")
            else:
                X = df[feature_cols].values
                y = df[target_col].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if task == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    try:
                        rmse = mean_squared_error(y_test, preds, squared=False)
                    except TypeError:
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                    st.success(f"R² Score: {r2:.4f} | RMSE: {rmse:.4f}")
                    result_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=result_df["Actual"],
                            y=result_df["Predicted"],
                            mode="markers",
                            marker=dict(size=8, color=hero_color["primary"], line=dict(width=1, color="#222")),
                            name="Predictions",
                        )
                    )
                    m, b = np.polyfit(result_df["Actual"], result_df["Predicted"], 1)
                    fig.add_trace(
                        go.Scatter(
                            x=result_df["Actual"],
                            y=m * result_df["Actual"] + b,
                            mode="lines",
                            name="Trend",
                            line=dict(color="#DC143C", width=3),
                        )
                    )
                    fig.update_layout(
                        title="Actual vs Predicted Values",
                        xaxis_title="Actual",
                        yaxis_title="Predicted",
                        plot_bgcolor="#181818",
                        paper_bgcolor="#18172a",
                        font=dict(color=hero_color["primary"]),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="lin_pred")

                    residuals = result_df["Actual"] - result_df["Predicted"]
                    res_fig = px.histogram(
                        residuals,
                        nbins=30,
                        title="Residuals Distribution",
                        labels={"value": "Residual", "count": "Frequency"},
                    )
                    res_fig.update_layout(
                        plot_bgcolor="#181818",
                        paper_bgcolor="#18172a",
                        font=dict(color=hero_color["primary"]),
                    )
                    st.plotly_chart(res_fig, use_container_width=True, key="lin_resid")

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="linear_regression_results.csv",
                        mime="text/csv",
                    )

                    message = INTERPRET_TEMPLATES[hero_choice][task].format(metric=r2)
                    typewriter([message], color=hero_color["primary"])
                    typewriter(hero["farewell_lines"], color=hero_color["primary"])

            else:
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = model.score(X_test, y_test)
                    st.success(f"Accuracy: {acc:.2f}")
                    cm = confusion_matrix(y_test, preds)
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="YlGnBu",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Class 0", "Class 1"],
                        y=["Class 0", "Class 1"],
                    )
                    fig.update_layout(
                        title="Confusion Matrix",
                        plot_bgcolor="#181818",
                        paper_bgcolor="#18172a",
                        font=dict(color=hero_color["primary"]),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="log_conf")

                    if len(np.unique(y_test)) == 2:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        roc_fig = go.Figure()
                        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
                        roc_fig.add_trace(
                            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random")
                        )
                        roc_fig.update_layout(
                            title=f"ROC Curve (AUC={roc_auc:.2f})",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            plot_bgcolor="#181818",
                            paper_bgcolor="#18172a",
                            font=dict(color=hero_color["primary"]),
                        )
                        st.plotly_chart(roc_fig, use_container_width=True, key="log_roc")

                    message = INTERPRET_TEMPLATES[hero_choice][task].format(metric=acc)
                    typewriter([message], color=hero_color["primary"])
                    typewriter(hero["farewell_lines"], color=hero_color["primary"])

    elif task == "K-Means Clustering":
        st.subheader("K-Means Clustering Setup")
        if len(numeric_cols) < 2:
            st.error("Dataset needs at least 2 numeric columns for K-Means!")
        else:
            feature_cols = st.multiselect(
                "Select 2 features for clustering:",
                numeric_cols,
                default=numeric_cols[:2],
                key="kmeans_features",
            )
            k = st.slider("Select number of clusters (K):", 2, 8, 3)
            if st.button("Run K-Means Clustering"):
                if len(feature_cols) != 2:
                    st.error("Please select exactly 2 features for 2D visualization.")
                else:
                    X = df[feature_cols].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    centers = kmeans.cluster_centers_

                    result_df = df.copy()
                    result_df["Cluster"] = clusters

                    st.success("Clustering complete! See plot and download results below.")
                    fig = px.scatter(
                        x=X_scaled[:, 0],
                        y=X_scaled[:, 1],
                        color=clusters.astype(str),
                        title="K-Means Clustering Results",
                        labels={"x": feature_cols[0], "y": feature_cols[1]},
                        symbol=clusters.astype(str),
                        height=500,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=centers[:, 0],
                            y=centers[:, 1],
                            mode="markers",
                            marker=dict(symbol="x", color="white", size=16, line=dict(width=3, color=hero_color["primary"])),
                            name="Centers",
                        )
                    )
                    fig.update_layout(
                        plot_bgcolor="#181818",
                        paper_bgcolor="#18172a",
                        font=dict(color=hero_color["primary"]),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="kmeans")
                    st.write("**Cluster Centers (Standardized Scale):**")
                    centers_df = pd.DataFrame(centers, columns=feature_cols)
                    st.dataframe(centers_df)
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Data with Clusters",
                        data=csv,
                        file_name="kmeans_results.csv",
                        mime="text/csv",
                    )

                    message = INTERPRET_TEMPLATES[hero_choice][task].format(k=k)
                    typewriter([message], color=hero_color["primary"])
                    typewriter(hero["farewell_lines"], color=hero_color["primary"])

    st.sidebar.markdown("---")
    st.sidebar.write("© Hall of Justice")