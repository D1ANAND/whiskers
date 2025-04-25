import asyncio
import csv
import json
import requests
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from mcp_agent.core.fastagent import FastAgent

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAgent
fast = FastAgent("LiquorRecommender")

# Define agents with OpenAI GPT models
@fast.agent(
    "analyze_bar_profile",
    instruction="Analyze user bar to extract key taste preferences, focusing on spirit types, ABV, and price range.",
    model="gpt-4o-mini"
)
@fast.agent(
    "enhanced_recommender",
    instruction="Recommend 5 bottles based on user preferences (favorite spirit and ABV). Provide convincing reasons why each bottle matches the user's taste profile. Use pre-filtered candidates and ensure recommendations are unique.",
    model="gpt-4o"
)
@fast.agent(
    "format_recommendations",
    instruction="Format the top 5 recommendations in JSON: { bottles: [{name, reason}] }. Ensure reasons are concise and convincing, explaining why each bottle suits the user.",
    model="gpt-4o-mini"
)

# Define the agent chain
@fast.chain(
    name="LiquorBartender",
    sequence=[
        "analyze_bar_profile",
        "enhanced_recommender",
        "format_recommendations"
    ]
)

# Helper functions
def analyze_user_bar(user_bar):
    """Analyze the user's bar to create a summary profile."""
    if not user_bar:
        return {
            'avg_proof': 80.0,
            'proof_range': [80.0, 80.0],
            'price_range': [30.0, 30.0],
            'spirits': ['whiskey'],
            'favorite_spirits': ['whiskey'],
            'favorite_brands': ['Jameson']
        }

    products = [b['product'] for b in user_bar]
    
    try:
        proofs = [float(p['proof']) for p in products]
        prices = [float(p['average_msrp']) for p in products]
        brands = [p['brand'] for p in products]
        spirits = [p['spirit'] for p in products]
    except (KeyError, ValueError) as e:
        print(f"Error processing user bar data: {e}")
        return {
            'avg_proof': 80.0,
            'proof_range': [80.0, 80.0],
            'price_range': [30.0, 30.0],
            'spirits': ['whiskey'],
            'favorite_spirits': ['whiskey'],
            'favorite_brands': ['Jameson']
        }
    
    min_proof = min(proofs) if proofs else 80.0
    max_proof = max(proofs) if proofs else 80.0
    avg_proof = sum(proofs) / len(proofs) if proofs else 80.0
    
    min_price = min(prices) if prices else 30.0
    max_price = max(prices) if prices else 30.0
    
    # Count spirits and brands using dictionary comprehension
    spirit_counts = {}
    for spirit in spirits:
        spirit_counts[spirit] = spirit_counts.get(spirit, 0) + 1
    
    brand_counts = {}
    for brand in brands:
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    favorite_spirits = sorted(spirit_counts.items(), key=lambda x: x[1], reverse=True)
    favorite_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'avg_proof': avg_proof,
        'proof_range': [min_proof, max_proof],
        'price_range': [min_price, max_price],
        'spirits': list(spirit_counts.keys()),
        'favorite_spirits': [s[0] for s in favorite_spirits] if favorite_spirits else ['whiskey'],
        'favorite_brands': [b[0] for b in favorite_brands] if favorite_brands else ['Jameson']
    }

def prefilter_liquors(dataset, favorite_spirit, target_abv, max_candidates=20):
    """Filter the dataset to get candidates matching the user's favorite spirit and ABV."""
    scored = []
    for item in dataset:
        score = 0
        try:
            item_abv = float(item["abv"])
            item_spirit = item["spirit_type"]
            item_price = float(item["shelf_price"])
            if target_abv - 5 <= item_abv <= target_abv + 5:
                score += 2
            if favorite_spirit.lower() == item_spirit.lower():
                score += 3
            if item_price <= 100:
                score += 1
            scored.append((score, item))
        except (ValueError, KeyError):
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_candidates]]

def load_liquor_dataset():
    """Load liquor dataset from CSV or use fallback."""
    dataset = []
    try:
        with open('liquors.csv', mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                dataset.append(row)
    except FileNotFoundError:
        print("Error: liquors.csv not found. Using sample dataset.")
        dataset = [
            {"name": "Jameson", "abv": 40, "spirit_type": "whiskey", "shelf_price": 30},
            {"name": "Maker's Mark", "abv": 45, "spirit_type": "whiskey", "shelf_price": 35},
            {"name": "Bulleit Bourbon", "abv": 45, "spirit_type": "whiskey", "shelf_price": 40},
            {"name": "Woodford Reserve", "abv": 45, "spirit_type": "whiskey", "shelf_price": 50},
            {"name": "Knob Creek", "abv": 50, "spirit_type": "whiskey", "shelf_price": 55},
            {"name": "Grey Goose", "abv": 40, "spirit_type": "vodka", "shelf_price": 45}
        ]
    return dataset

async def fetch_user_bar(username: str):
    """Fetch user bar data from API or use fallback."""
    user_bar_url = f"http://services.baxus.co/api/bar/user/{username}"
    try:
        response = requests.get(user_bar_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching user bar for {username}: {e}")
        return [
            {"product": {"name": "Jameson", "proof": 80, "average_msrp": 30, "brand": "Jameson", "spirit": "whiskey"}}
        ]

async def aggregate_profiles(profiles: list[dict]):
    """Aggregate multiple user profiles into a combined profile."""
    if not profiles:
        raise ValueError("No profiles provided for aggregation.")
    
    all_spirits = []
    for profile in profiles:
        all_spirits.extend(profile['favorite_spirits'])
    
    spirit_counts = {spirit: all_spirits.count(spirit) for spirit in set(all_spirits)}
    favorite_spirit = max(spirit_counts.items(), key=lambda x: x[1])[0] if spirit_counts else "whiskey"
    
    avg_abvs = [profile['avg_proof'] / 2 for profile in profiles]
    avg_abv = sum(avg_abvs) / len(avg_abvs) if avg_abvs else 40.0
    
    min_prices = [profile['price_range'][0] for profile in profiles]
    max_prices = [profile['price_range'][1] for profile in profiles]
    price_range = [min(min_prices), max(max_prices)]
    
    return {
        'avg_proof': avg_abv * 2,
        'price_range': price_range,
        'favorite_spirits': [favorite_spirit]
    }

async def find_influential_user(bottle: dict, profiles: list[dict], usernames: list[str]):
    """Determine which user's profile most influenced a bottle recommendation."""
    bottle_spirit = bottle.get('spirit_type', '').lower()
    bottle_abv = float(bottle.get('abv', 0))
    
    max_score = -1
    influential_user = usernames[0]
    
    for profile, username in zip(profiles, usernames):
        score = 0
        favorite_spirit = profile['favorite_spirits'][0].lower() if profile['favorite_spirits'] else ''
        avg_abv = profile['avg_proof'] / 2
        
        if bottle_spirit == favorite_spirit:
            score += 3
        if abs(bottle_abv - avg_abv) <= 5:
            score += 2
            
        if score > max_score:
            max_score = score
            influential_user = username
    
    return influential_user

async def run_recommendation_pipeline(username: str, dataset: list[dict], fast: FastAgent):
    """Run the recommendation pipeline for a single user."""
    user_data = await fetch_user_bar(username)
    user_profile = analyze_user_bar(user_data)
    favorite_spirit = user_profile['favorite_spirits'][0] if user_profile['favorite_spirits'] else "whiskey"
    avg_abv = user_profile['avg_proof'] / 2
    
    candidate_bottles = prefilter_liquors(dataset, favorite_spirit, avg_abv, max_candidates=20)
    if len(candidate_bottles) < 5:
        return {"error": f"Insufficient candidates ({len(candidate_bottles)}) for recommendations."}
    
    async with fast.run() as agent:
        profile_result = await agent.analyze_bar_profile(user_profile)
        recommendations = await agent.enhanced_recommender(candidate_bottles, profile_result)
        formatted_results = await agent.format_recommendations(recommendations)
        return formatted_results

async def run_room_recommendation_pipeline(usernames: list[str], dataset: list[dict], fast: FastAgent):
    """Run the recommendation pipeline for multiple users."""
    user_profiles = []
    for username in usernames:
        user_data = await fetch_user_bar(username)
        profile = analyze_user_bar(user_data)
        user_profiles.append(profile)
    
    combined_profile = await aggregate_profiles(user_profiles)
    favorite_spirit = combined_profile['favorite_spirits'][0]
    avg_abv = combined_profile['avg_proof'] / 2
    
    candidate_bottles = prefilter_liquors(dataset, favorite_spirit, avg_abv, max_candidates=20)
    if len(candidate_bottles) < 5:
        return {"error": f"Insufficient candidates ({len(candidate_bottles)}) for recommendations."}
    
    async with fast.run() as agent:
        profile_result = await agent.analyze_bar_profile(combined_profile)
        recommendations = await agent.enhanced_recommender(candidate_bottles, profile_result)
        formatted_results = await agent.format_recommendations(recommendations)
    
    influenced_by = []
    for bottle in formatted_results['bottles']:
        bottle_info = next((item for item in dataset if item['name'] == bottle['name']), {})
        influential_user = await find_influential_user(bottle_info, user_profiles, usernames)
        influenced_by.append({"bottle": bottle['name'], "influenced_by": influential_user})
    
    return {
        "bottles": formatted_results['bottles'],
        "influenced_by": influenced_by
    }

# Flask Endpoints
@app.route('/personalized-recommendations', methods=['POST'])
def personalized_recommendations():
    """Endpoint for personalized liquor recommendations."""
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({"error": "Username is required."}), 400
    
    dataset = load_liquor_dataset()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_recommendation_pipeline(username, dataset, fast))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()

@app.route('/room-recommendations', methods=['POST'])
def room_recommendations():
    """Endpoint for room-based liquor recommendations."""
    data = request.get_json()
    usernames = data.get('usernames')
    if not usernames or not isinstance(usernames, list) or len(usernames) < 1:
        return jsonify({"error": "At least one username is required as a list."}), 400
    
    dataset = load_liquor_dataset()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_room_recommendation_pipeline(usernames, dataset, fast))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)