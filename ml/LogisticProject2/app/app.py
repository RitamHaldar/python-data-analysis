from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

try:
	import joblib
except Exception:
	joblib = None


st.set_page_config(
	page_title="Adherence Intelligence",
	page_icon="AI",
	layout="wide",
	initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Libre+Baskerville:wght@700&display=swap');

		:root {
			--bg: #f7f1e4;
			--bg-soft: #fff9ed;
			--ink: #1f2a37;
			--muted: #5b6778;
			--accent: #0b6e6e;
			--accent-soft: #d5f0ec;
			--gold: #b68a2a;
			--card: rgba(255, 255, 255, 0.78);
			--border: rgba(11, 110, 110, 0.20);
		}

		.stApp {
			background:
			  radial-gradient(1000px 500px at 0% 0%, #fff5d9 0%, rgba(255,245,217,0) 62%),
			  radial-gradient(850px 550px at 100% 0%, #dff5f2 0%, rgba(223,245,242,0) 62%),
			  linear-gradient(180deg, var(--bg-soft) 0%, var(--bg) 100%);
			color: var(--ink);
			font-family: 'Manrope', sans-serif;
		}

		[data-testid="stSidebar"] {
			background: linear-gradient(180deg, #f4ead5 0%, #efe4cc 100%);
			border-right: 1px solid rgba(182, 138, 42, 0.25);
		}

		.hero {
			background: linear-gradient(120deg, rgba(11,110,110,0.92), rgba(33,109,124,0.90));
			padding: 1.4rem 1.6rem;
			border-radius: 18px;
			color: #ffffff;
			box-shadow: 0 18px 40px rgba(15, 56, 63, 0.25);
			border: 1px solid rgba(255,255,255,0.20);
			margin-bottom: 1rem;
			position: relative;
			overflow: hidden;
		}

		.hero::after {
			content: "";
			position: absolute;
			width: 220px;
			height: 220px;
			border-radius: 50%;
			right: -55px;
			top: -65px;
			background: radial-gradient(circle, rgba(255,255,255,0.28), rgba(255,255,255,0));
		}

		.hero h1 {
			font-family: 'Libre Baskerville', serif;
			letter-spacing: 0.3px;
			font-size: 2rem;
			margin: 0;
		}

		.hero p {
			margin-top: 0.55rem;
			margin-bottom: 0;
			max-width: 900px;
			color: rgba(255,255,255,0.92);
			font-size: 1rem;
		}

		.glass-card {
			background: var(--card);
			backdrop-filter: blur(10px);
			border: 1px solid var(--border);
			border-radius: 16px;
			padding: 0.9rem 1rem;
			box-shadow: 0 10px 25px rgba(56, 69, 84, 0.12);
		}

		.section-title {
			margin-top: 0.2rem;
			margin-bottom: 0.4rem;
			color: var(--ink);
			font-weight: 800;
			font-size: 1.06rem;
		}

		.tiny {
			color: var(--muted);
			font-size: 0.88rem;
			margin-top: 0;
		}

		div[data-testid="stMetricValue"] {
			color: var(--accent);
		}

		.stButton > button {
			border-radius: 11px;
			border: none;
			background: linear-gradient(135deg, var(--gold), #a2781f);
			color: #fff;
			font-weight: 700;
			padding: 0.55rem 1.1rem;
			box-shadow: 0 10px 18px rgba(166, 125, 37, 0.26);
		}

		.stButton > button:hover {
			transform: translateY(-1px);
			filter: brightness(1.03);
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _load_serialized(path: Path):
	if joblib is not None:
		try:
			return joblib.load(path)
		except Exception:
			pass
	with path.open("rb") as file:
		return pickle.load(file)


@st.cache_resource
def load_artifacts():
	base = Path(__file__).parent
	model_path = base / "patient_adherence_model.pkl"
	scaler_path = base / "patient_datascaler.pkl"

	if not model_path.exists() or not scaler_path.exists():
		missing = []
		if not model_path.exists():
			missing.append(model_path.name)
		if not scaler_path.exists():
			missing.append(scaler_path.name)
		raise FileNotFoundError(f"Missing required artifact(s): {', '.join(missing)}")

	model = _load_serialized(model_path)
	scaler = _load_serialized(scaler_path)
	return model, scaler, model_path, scaler_path


def infer_feature_names(model, scaler):
	if hasattr(model, "feature_names_in_"):
		return list(model.feature_names_in_)
	if hasattr(scaler, "feature_names_in_"):
		return list(scaler.feature_names_in_)

	n_features = None
	if hasattr(scaler, "n_features_in_"):
		n_features = int(scaler.n_features_in_)
	elif hasattr(model, "n_features_in_"):
		n_features = int(model.n_features_in_)

	if n_features is None or n_features <= 0:
		n_features = 5

	return [f"feature_{i + 1}" for i in range(n_features)]


def pretty_label(name: str) -> str:
	return name.replace("_", " ").strip().title()


_inject_styles()

st.markdown(
	"""
	<div class="hero">
	  <h1>Adherence Intelligence Dashboard</h1>
	  <p>
		Enter patient features, apply your trained scaler, and generate model-based adherence predictions with confidence scoring.
	  </p>
	</div>
	""",
	unsafe_allow_html=True,
)

try:
	model, scaler, model_path, scaler_path = load_artifacts()
except Exception as exc:
	st.error(f"Artifact loading failed: {exc}")
	st.stop()

feature_names = infer_feature_names(model, scaler)

with st.sidebar:
	st.markdown("### Runtime Status")
	st.success("Model and scaler loaded")
	st.caption(f"Model: {model_path.name}")
	st.caption(f"Scaler: {scaler_path.name}")
	st.markdown("---")
	st.markdown("### Notes")
	st.write(
		"Input values are transformed using the scaler before prediction to match training-time preprocessing."
	)

left, right = st.columns([1.25, 1])

with left:
	st.markdown('<div class="glass-card">', unsafe_allow_html=True)
	st.markdown('<div class="section-title">Patient Feature Inputs</div>', unsafe_allow_html=True)
	st.markdown(
		'<p class="tiny">Provide numeric values for each feature used by your trained pipeline.</p>',
		unsafe_allow_html=True,
	)

	with st.form("prediction_form"):
		c1, c2 = st.columns(2)
		feature_values = {}
		for idx, feature in enumerate(feature_names):
			target_col = c1 if idx % 2 == 0 else c2
			with target_col:
				feature_values[feature] = st.number_input(
					pretty_label(feature),
					value=0.0,
					step=0.1,
					format="%.4f",
					key=f"feat_{idx}",
				)

		submitted = st.form_submit_button("Predict Adherence")
	st.markdown("</div>", unsafe_allow_html=True)

with right:
	st.markdown('<div class="glass-card">', unsafe_allow_html=True)
	st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
	st.markdown(
		'<p class="tiny">Run a prediction to see class output and confidence score.</p>',
		unsafe_allow_html=True,
	)

	if submitted:
		try:
			feature_vector = np.array([[feature_values[name] for name in feature_names]], dtype=float)
			scaled_vector = scaler.transform(feature_vector)

			prediction = model.predict(scaled_vector)[0]

			confidence = None
			if hasattr(model, "predict_proba"):
				probs = model.predict_proba(scaled_vector)
				if probs.ndim == 2 and probs.shape[1] > 1:
					confidence = float(np.max(probs[0]))
				else:
					confidence = float(probs.ravel()[0])

			label_map = {0: "Low Adherence", 1: "High Adherence"}
			readable_prediction = label_map.get(prediction, str(prediction))

			st.metric("Predicted Class", readable_prediction)
			if confidence is not None:
				st.metric("Confidence", f"{confidence * 100:.2f}%")
				st.progress(min(max(confidence, 0.0), 1.0))

			st.markdown("---")
			st.dataframe(pd.DataFrame([feature_values]), use_container_width=True)

		except Exception as exc:
			st.error(f"Prediction failed: {exc}")
	else:
		st.info("Submit the form to generate a prediction.")

	st.markdown("</div>", unsafe_allow_html=True)
