import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import plotly
import plotly.graph_objects as go
import json

from config import MODEL_PATH, TOKENIZER_PATH, DATA_PATH, SECRET_KEY


# --------------------------------------------------------
# NLTK DOWNLOADS
# --------------------------------------------------------
nltk_packages = ['wordnet', 'stopwords']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)


# --------------------------------------------------------
# FLASK APP INIT
# --------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY


# --------------------------------------------------------
# LOAD MODEL + VECTORIZER
# --------------------------------------------------------
try:
    vectorizer = joblib.load(TOKENIZER_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {e}")


# --------------------------------------------------------
# NLP UTILITIES
# --------------------------------------------------------
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean + preprocess text exactly like during training."""
    if not text:
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()

    words = text.split()
    words = [w for w in words if w not in stop]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# --------------------------------------------------------
# SYMPTOM → NATURAL LANGUAGE MAPPING
# --------------------------------------------------------
SYMPTOM_TEXT = {
    "Acne": "I have pimples, whiteheads, and red bumps on my skin.",
    "Anxiety": "I feel nervous, restless, and constantly worried.",
    "Arthritis": "My joints feel stiff, swollen, and painful.",
    "Asthma": "I have difficulty breathing and frequent wheezing episodes.",
    "Back pain": "I am experiencing persistent pain in my lower or upper back.",
    "Bipolar disorder": "My mood shifts between extreme highs and deep lows.",
    "Birth control": "I need guidance in choosing the best birth control method.",
    "Bronchitis": "I have a persistent cough and chest congestion.",
    "Chronic pain": "I have long-term body pain that doesn't go away.",
    "Cold": "I have a runny nose, sneezing, and mild fever.",
    "Constipation": "I have difficulty passing stools and feel bloated.",
    "Cough": "I am experiencing continuous coughing.",
    "Depression": "I feel sad, hopeless, and lack interest in activities.",
    "Diabetes": "I have high blood sugar levels and feel fatigued.",
    "Diarrhea": "I have frequent loose or watery bowel movements.",
    "Eczema": "My skin is dry, itchy, and inflamed.",
    "Fatigue": "I feel extremely tired and lack energy.",
    "Fever": "I have a high temperature, chills, and weakness.",
    "Flu": "I have fever, body aches, and severe fatigue.",
    "Gastroesophageal reflux disease (GERD)": "I feel acid reflux and burning pain in my chest.",
    "Headache": "I have continuous or throbbing pain in my head.",
    "High blood pressure": "My blood pressure is elevated and causing discomfort.",
    "High cholesterol": "I have high cholesterol levels and need management.",
    "Insomnia": "I have trouble falling asleep or staying asleep.",
    "Migraine": "I have intense headaches with light or sound sensitivity.",
    "Nausea": "I feel like vomiting and have stomach discomfort.",
    "Obesity": "I am struggling with excess weight and related symptoms.",
    "Pain": "I am experiencing persistent pain in my body.",
    "Pneumonia": "I have chest pain, cough, fever, and difficulty breathing.",
    "Psoriasis": "I have red, scaly patches on my skin.",
    "Sinusitis": "I have sinus pressure, blocked nose, and facial pain.",
    "Skin rash": "I have an itchy or irritated skin rash.",
    "Stress": "I feel overwhelmed, tense, and mentally exhausted.",
    "Thyroid disorder": "I have symptoms related to thyroid imbalance.",
    "Urinary tract infection (UTI)": "I feel burning while urinating and need to go frequently.",
    "Vomiting": "I have nausea and episodes of vomiting.",
    "Weight loss": "I am losing weight unexpectedly.",
    "Allergies": "I have sneezing, itching, and allergic reactions.",
    "Bladder infection": "I have pelvic pain and burning while urinating.",
    "Chest pain": "I feel tightness or discomfort in my chest.",
    "Dizziness": "I feel lightheaded and unbalanced.",
    "Ear infection": "I have ear pain and trouble hearing.",
    "Eye infection": "My eye is red, irritated, and watery.",
    "Fibromyalgia": "I have widespread pain and fatigue.",
    "Heartburn": "I feel burning pain in my chest after eating.",
    "Hemorrhoids": "I have pain or bleeding during bowel movements.",
    "Indigestion": "I feel discomfort or heaviness after eating.",
    "Irritable bowel syndrome (IBS)": "I have abdominal pain, bloating, and irregular bowel movements.",
    "Joint pain": "My joints hurt and feel stiff.",
    "Kidney stones": "I have sharp pain in my back or lower abdomen.",
    "Muscle pain": "My muscles are sore or achy.",
    "Seasonal allergies": "I have sneezing, itching, and a runny nose during certain seasons.",
    "Sore throat": "It hurts when I swallow or talk.",
    "Swelling": "A part of my body is swollen or inflamed.",
    "Toothache": "I have severe pain in one of my teeth.",
    "Vaginal infection": "I have itching, discharge, or discomfort in the vaginal area.",
    "Wheezing": "I hear a whistling sound when I breathe.",
}


# --------------------------------------------------------
# BACKUP MEDICINES
# --------------------------------------------------------
BACKUP_MEDICINES = {
    "fever": ["Paracetamol", "Ibuprofen", "Acetaminophen"],
    "cold": ["Cetirizine", "Levocetirizine", "Phenylephrine"],
    "cough": ["Dextromethorphan", "Guaifenesin", "Bromhexine"],
    "headache": ["Ibuprofen", "Aspirin", "Naproxen"],
    "sore throat": ["Azithromycin", "Amoxicillin", "Chlorhexidine gargle"],
    "nausea": ["Ondansetron", "Domperidone", "Promethazine"],
    "vomiting": ["Ondansetron", "Domperidone", "Promethazine"],
    "diarrhea": ["Loperamide", "Oral Rehydration Salts (ORS)", "Rifaximin"],
    "constipation": ["Lactulose", "Bisacodyl", "Polyethylene glycol"],
    "pain": ["Ibuprofen", "Paracetamol", "Diclofenac"],
    "back pain": ["Ibuprofen", "Paracetamol", "Cyclobenzaprine"],
    "migraine": ["Sumatriptan", "Ibuprofen", "Naproxen"],
    "allergies": ["Cetirizine", "Loratadine", "Fexofenadine"],
    "urinary tract infection (uti)": ["Nitrofurantoin", "Ciprofloxacin", "Trimethoprim"],
    "pneumonia": ["Amoxicillin", "Azithromycin", "Levofloxacin"],
    "__default__": ["Paracetamol", "Ibuprofen", "Cetirizine"]
}


# --------------------------------------------------------
# DIRECT SYMPTOM MATCH
# --------------------------------------------------------
def match_symptom_keyword(text):
    text = text.lower()
    for symptom in SYMPTOM_TEXT.keys():
        if symptom.lower() in text:
            return symptom
    return None


# --------------------------------------------------------
# GET DRUGS
# --------------------------------------------------------
def get_drugs_from_df_or_backup(condition, top_n=3):
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        key = condition.lower() if isinstance(condition, str) else ""
        return BACKUP_MEDICINES.get(key, BACKUP_MEDICINES["__default__"])[:top_n]

    if "condition" not in df.columns or "drugName" not in df.columns:
        key = condition.lower() if isinstance(condition, str) else ""
        return BACKUP_MEDICINES.get(key, BACKUP_MEDICINES["__default__"])[:top_n]

    df["condition_lower"] = df["condition"].astype(str).str.lower()
    cond_lower = str(condition).lower()

    matches = df[df["condition_lower"] == cond_lower]

    if not matches.empty:
        matches = matches.sort_values(by="rating", ascending=False) if "rating" in matches.columns else matches
        drugs = matches["drugName"].head(top_n).tolist()
        seen = set()
        uniq = []
        for d in drugs:
            if d not in seen:
                uniq.append(d)
                seen.add(d)
        if len(uniq) >= 1:
            return uniq

    return BACKUP_MEDICINES.get(cond_lower, BACKUP_MEDICINES["__default__"])[:top_n]


# --------------------------------------------------------
# ML PREDICTION
# --------------------------------------------------------
def predict_condition_and_drugs(raw_text):
    processed = preprocess_text(raw_text)
    tfidf = vectorizer.transform([processed])
    pred = model.predict(tfidf)[0]
    top_drugs = get_drugs_from_df_or_backup(pred, top_n=3)
    return pred, top_drugs


# --------------------------------------------------------
# GRAPH
# --------------------------------------------------------
def create_enhanced_chart(top_drugs):
    if not top_drugs:
        return None

    drugs_reversed = list(reversed(top_drugs))
    ranks_reversed = list(reversed(list(range(1, len(top_drugs) + 1))))

    base_colors = ['#667eea', '#4facfe', '#f093fb', '#f38ba8', '#a6e3a1']
    colors_reversed = list(reversed(base_colors[:len(top_drugs)]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=drugs_reversed,
        x=ranks_reversed,
        orientation='h',
        marker=dict(
            color=colors_reversed,
            line=dict(color='rgba(255, 255, 255, 0.12)', width=1)
        ),
        text=ranks_reversed,
        textposition='inside',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate='<b>%{y}</b><br>Rank: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': 'Drug Recommendation Ranking', 'font': {'size': 20, 'color': '#ffffff'}, 'x': 0.5},
        xaxis=dict(title='Rank (1 = Best)', gridcolor='rgba(255,255,255,0.06)', color='#b4b4c8', dtick=1),
        yaxis=dict(title='Drug Name', gridcolor='rgba(255,255,255,0.03)', color='#b4b4c8'),
        plot_bgcolor='rgba(255,255,255,0.02)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=380,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig


# --------------------------------------------------------
# ROUTES
# --------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == "POST":

        input_method = request.form.get("input_method")

        # ---------- PREDEFINED SYMPTOMS ----------
        if input_method == "predefined":
            selected = request.form.getlist("symptoms")

            if not selected:
                flash("Please select at least one symptom.", "warning")
                return redirect(url_for('index'))

            pred = selected[0]
            raw_text = " ".join(SYMPTOM_TEXT.get(s, s) for s in selected)
            top_drugs = get_drugs_from_df_or_backup(pred, top_n=3)

        # ---------- FREE TEXT ----------
        else:
            raw_text = request.form.get("free_text", "")

            if not raw_text.strip():
                flash("Please describe your symptoms.", "warning")
                return redirect(url_for('index'))

            matched = match_symptom_keyword(raw_text)

            # ⭐ NEW RULE: If no disease is found → NO DRUGS, NO GRAPH
            if not matched:
                return render_template(
                    "results.html",
                    predicted="No disease found",
                    top_drugs=[],
                    graphJSON=None,
                    raw_text=raw_text
                )

            pred = matched
            top_drugs = get_drugs_from_df_or_backup(pred, top_n=3)

        # ---------- GRAPH ----------
        fig = create_enhanced_chart(top_drugs)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) if fig else None

        return render_template("results.html", predicted=pred, top_drugs=top_drugs, graphJSON=graphJSON, raw_text=raw_text)

    return render_template("index.html", symptoms=list(SYMPTOM_TEXT.keys()))


# --------------------------------------------------------
# API
# --------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    matched = match_symptom_keyword(text)

    if not matched:
        return jsonify({"predicted": "No disease found", "top_drugs": []})

    pred = matched
    top_drugs = get_drugs_from_df_or_backup(pred, top_n=3)

    return jsonify({"predicted": pred, "top_drugs": top_drugs})


# --------------------------------------------------------
# RUN APP
# --------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
