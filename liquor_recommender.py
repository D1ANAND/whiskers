import asyncio
import csv
import json
import requests
import sys
import os
from dotenv import load_dotenv
from mcp_agent.core.fastagent import FastAgent

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

def analyze_user_bar(user_bar):
    """Analyze the user's bar to create a summary profile."""
    products = [b['product'] for b in user_bar]
    
    proofs = [float(p['proof']) for p in products]
    prices = [float(p['average_msrp']) for p in products]
    brands = [p['brand'] for p in products]
    spirits = [p['spirit'] for p in products]
    
    min_proof = min(proofs)
    max_proof = max(proofs)
    avg_proof = sum(proofs) / len(proofs)
    
    min_price = min(prices)
    max_price = max(prices)
    
    spirit_counts = {}
    spirit_counts = {spirit: spirit_counts.get(spirit, 0) + 1 for spirit in spirits}
    brand_counts = {}
    brand_counts = {brand: brand_counts.get(brand, 0) + 1 for brand in brands}
    
    favorite_spirits = sorted(spirit_counts.items(), key=lambda x: x[1], reverse=True)
    favorite_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'avg_proof': avg_proof,
        'proof_range': [min_proof, max_proof],
        'price_range': [min_price, max_price],
        'spirits': list(spirit_counts.keys()),
        'favorite_spirits': [s[0] for s in favorite_spirits],
        'favorite_brands': [b[0] for b in favorite_brands]
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

async def main():
    # Get username from command-line argument or environment variable
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = os.getenv("BAXUS_USERNAME")
        if not username:
            print("Error: Username not provided. Run with `python liquor_recommender.py username` or set BAXUS_USERNAME in .env")
            return

    # Fetch user bar data
    user_bar_url = f"http://services.baxus.co/api/bar/user/{username}"
    try:
        response = requests.get(user_bar_url, timeout=5)
        response.raise_for_status()
        user_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching user bar for {username}: {e}")
        # Fallback to sample data
        user_data = [
            {"product": {"name": "Jameson", "proof": 80, "average_msrp": 30, "brand": "Jameson", "spirit": "whiskey"}}
        ]

    # Load liquor dataset
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
            {"name": "Grey Goose", "abv": 40, "spirit_type": "vodka", "shelf_price": 45}
        ]

    # Analyze user bar
    user_profile = analyze_user_bar(user_data)
    favorite_spirit = user_profile['favorite_spirits'][0] if user_profile['favorite_spirits'] else "whiskey"
    avg_abv = user_profile['avg_proof'] / 2  # Convert proof to ABV

    # Filter dataset
    candidate_bottles = prefilter_liquors(dataset, favorite_spirit, avg_abv, max_candidates=20)

    # Run FastAgent pipeline
    async with fast.run() as agent:
        profile_result = await agent.analyze_bar_profile(user_profile)
        recommendations = await agent.enhanced_recommender(candidate_bottles, profile_result)
        formatted_results = await agent.format_recommendations(recommendations)
        print(json.dumps(formatted_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())