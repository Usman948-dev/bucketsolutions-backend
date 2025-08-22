# app.py (Modified to rebuild models on Render startup)
from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import joblib # Still needed for joblib.dump if you want to save after upload
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app) 

DATA_FILE_PATH = "Sample-Superstore.csv" 
MODELS_DIR = "models" # This folder will be created on Render

# Global variables (will be populated on startup)
sales_df = None
tfidf_matrix = None
cosine_sim = None
indices = None
association_rules_df = None
user_item_matrix = None
most_popular_model = None

# --- Data Preprocessing and Model Building Functions (keep these) ---
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    required_columns = ['Order ID', 'Order Date', 'Customer ID', 'Product Name', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Profit']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
    df.dropna(subset=['Sales', 'Profit', 'Product Name', 'Category', 'Customer ID', 'Order ID'], inplace=True)
    return df

def build_market_basket_model(df):
    basket = (df.groupby(['Order ID', 'Product Name'])['Product Name']
              .count().unstack().reset_index().fillna(0)
              .set_index('Order ID'))
    def encode_units(x):
        return 1 if x > 0 else 0
    basket_encoded = basket.applymap(encode_units)
    try:
        frequent_itemsets = fpgrowth(basket_encoded, min_support=0.001, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        return rules
    except (ValueError, MemoryError) as e:
        print(f"Warning: Market Basket Analysis failed. {e}. Returning empty rules.")
        return pd.DataFrame()

def build_content_based_model(df):
    df_cb = df.drop_duplicates(subset='Product Name')
    df_cb.set_index('Product Name', inplace=True)
    df_cb['combined_features'] = df_cb['Category'].fillna('') + ' ' + df_cb['Sub-Category'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_cb['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df_cb.index)
    return {'cosine_sim': cosine_sim, 'indices': indices}

def build_collaborative_filtering_model(df):
    user_item = df.groupby(['Customer ID', 'Product Name'])['Quantity'].sum().unstack().fillna(0)
    return user_item

def build_most_popular_model(df):
    most_popular = df.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False).head(10)
    return most_popular

# --- Model Initialization Logic (Rebuilds models on app startup) ---
@app.before_first_request
def initialize_models(): # Renamed from load_models to reflect rebuilding
    """Builds all models from Sample-Superstore.csv on Flask app startup."""
    global sales_df, tfidf_matrix, cosine_sim, indices, association_rules_df, user_item_matrix, most_popular_model
    print("--- Initializing models on Flask startup (rebuilding from CSV) ---")
    try:
        if not os.path.exists(DATA_FILE_PATH):
            print(f"Data file '{DATA_FILE_PATH}' not found. Cannot build models.")
            sales_df = pd.DataFrame()
            association_rules_df = pd.DataFrame()
            content_based_model_data = {'cosine_sim': None, 'indices': None}
            user_item_matrix = pd.DataFrame()
            most_popular_model = pd.Series()
            return

        df = pd.read_csv(DATA_FILE_PATH, encoding='ISO-8859-1')
        sales_df = preprocess_data(df)
        products_df = sales_df.drop_duplicates(subset='Product Name')

        # Create models directory if it doesn't exist (Render's ephemeral storage)
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        print("Building Market Basket model...")
        association_rules_df = build_market_basket_model(sales_df)

        print("Building Content-Based model...")
        content_based_model_data = build_content_based_model(sales_df)
        cosine_sim = content_based_model_data['cosine_sim']
        indices = content_based_model_data['indices']

        print("Building Collaborative Filtering model...")
        user_item_matrix = build_collaborative_filtering_model(sales_df)

        print("Building Most Popular model...")
        most_popular_model = build_most_popular_model(sales_df)

        print("--- All models built and loaded successfully from CSV! ---")
    except Exception as e:
        print(f"An error occurred during model building on startup: {e}")
        sales_df = pd.DataFrame()
        association_rules_df = pd.DataFrame()
        content_based_model_data = {'cosine_sim': None, 'indices': None}
        user_item_matrix = pd.DataFrame()
        most_popular_model = pd.Series()

# --- Flask Endpoints (same as before) ---
@app.route('/', methods=['GET'])
def home():
    return "<h1>Recommendation Engine API is Running!</h1>"

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint to upload data from front-end and trigger model building."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded file to Render's ephemeral storage
        file.save(DATA_FILE_PATH)
        # Rebuild models after new file upload
        initialize_models() # Call the main initialization function
        return jsonify({"message": "File uploaded and models built successfully."})
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        return jsonify({"error": f"Error processing uploaded file: {e}"}), 500

@app.route('/recommend/basket', methods=['POST'])
def get_basket_recommendations():
    if association_rules_df.empty:
        return jsonify({"recommendations": []}), 200

    items_in_cart = request.json.get('items', [])
    if not items_in_cart:
        return jsonify({"error": "No items provided in cart"}), 400

    recommendations = set()
    for item in items_in_cart:
        item_frozenset = frozenset([item])
        if 'antecedents' in association_rules_df.columns:
            rules = association_rules_df[association_rules_df['antecedents'].apply(lambda x: item_frozenset.issubset(x))]
            for _, row in rules.iterrows():
                recommendations.update(row['consequents'])
        else:
            return jsonify({"recommendations": []}), 200

    return jsonify({"recommendations": list(recommendations)})

@app.route('/recommend/content', methods=['POST'])
def get_content_recommendations():
    if sales_df.empty or cosine_sim is None or indices is None:
        return jsonify({"error": "Content-Based model not built. Please check logs."}), 404
    product_name = request.json.get('product_name')
    if not product_name:
        return jsonify({"error": "No product name provided"}), 400
    if product_name not in indices.values:
        return jsonify({"error": f"Product '{product_name}' not found in the content-based model."}), 404
    idx = indices[indices == product_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    item_indices = [i[0] for i in sim_scores]
    recommendations = sales_df.iloc[item_indices]['Product Name'].tolist()
    return jsonify({"recommendations": recommendations})

@app.route('/recommend/collaborative', methods=['POST'])
def get_collaborative_recommendations():
    if user_item_matrix.empty:
        return jsonify({"error": "Collaborative Filtering model not built. Please check logs."}), 404
    customer_id = request.json.get('customer_id')
    if not customer_id:
        return jsonify({"error": "No customer ID provided"}), 400
    if customer_id not in user_item_matrix.index:
        return jsonify({"error": f"Customer '{customer_id}' not found in the collaborative filtering model."}), 404
    user_purchases = user_item_matrix.loc[customer_id]
    if len(user_item_matrix) > 1:
        similar_users = user_item_matrix.drop(customer_id).corrwith(user_purchases, axis=1).sort_values(ascending=False)
        similar_users = similar_users.dropna()
    else:
        similar_users = pd.Series([])
    recommendations = []
    if not similar_users.empty:
        most_similar_user_id = similar_users.index[0]
        similar_user_purchases = user_item_matrix.loc[most_similar_user_id][user_item_matrix.loc[most_similar_user_id] > 0]
        user_bought = user_purchases[user_purchases > 0].index
        recommendations = [prod for prod in similar_user_purchases.index if prod not in user_bought]
    return jsonify({"recommendations": recommendations[:5]})

@app.route('/recommend/popular', methods=['GET'])
def get_most_popular_recommendations():
    if most_popular_model.empty:
        return jsonify({"error": "Most popular model not built. Please check logs."}), 404
    recommendations = most_popular_model.index.tolist()
    return jsonify({"recommendations": recommendations})

# NEW: Hybrid Recommendation Endpoint
@app.route('/recommend/hybrid', methods=['POST'])
def get_hybrid_recommendations():
    customer_id = request.json.get('customer_id')
    if not customer_id:
        return jsonify({"error": "No customer ID provided for hybrid recommendation"}), 400

    hybrid_recs = set()

    # 1. Get Collaborative Filtering recommendations
    collab_response = get_collaborative_recommendations() 
    if collab_response.status_code == 200:
        collab_data = json.loads(collab_response.get_data(as_text=True))
        hybrid_recs.update(collab_data.get('recommendations', []))
    else:
        print(f"Warning: Collaborative Filtering failed for hybrid recs: {collab_response.get_data(as_text=True)}")

    # 2. Get Most Popular recommendations (as a fallback or diversity)
    popular_response = get_most_popular_recommendations() 
    if popular_response.status_code == 200:
        popular_data = json.loads(popular_response.get_data(as_text=True))
        if len(hybrid_recs) < 5: 
            hybrid_recs.update(popular_data.get('recommendations', [])[:5]) 
    else:
        print(f"Warning: Most Popular recommendations failed for hybrid recs: {popular_response.get_data(as_text=True)}")

    return jsonify({"recommendations": list(hybrid_recs)[:10]}) 

# Flask endpoint to serve the HTML dashboard (Render will serve static files)
@app.route('/dashboard', methods=['GET'])
def serve_dashboard():
    return "Front-end is served from a separate static site!"

# No app.run() here. Gunicorn (via Procfile) will run the app.
