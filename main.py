from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import requests
import datetime
import os
import logging
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, BigInteger, Boolean, DateTime, String, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# Environment variables (Replace with your actual values)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key here
DATABASE_URL = os.getenv("DATABASE_URL")  # Get database URL from environment variable
MCP_FILE_ID = os.getenv("MCP_FILE_ID")  # Get file ID from environment variable (this is your uploaded CSV file)

# Database setup
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")
engine = create_engine(DATABASE_URL, echo=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI instance
app = FastAPI()

# CORS setup for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# SQLAlchemy models
class Bottle(Base):
    __tablename__ = "bottles"
    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String)
    spirit_type = Column(String)
    fair_price = Column(Float)
    abv = Column(Float)
    ranking = Column(Integer)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)

class UserRating(Base):
    __tablename__ = "user_ratings"
    user_id = Column(String, primary_key=True, index=True)  # Changed to string for username
    bottle_id = Column(BigInteger, primary_key=True, index=True)
    liked = Column(Boolean)
    rated_at = Column(DateTime, default=datetime.datetime.utcnow)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request model for user ratings
class RatingModel(BaseModel):
    username: str  # Changed to use username as a string
    bottle_id: int
    liked: bool

# Function to fetch user bar data from BAXUS API using username (string)
def get_user_bar_data(username: str):
    url = f"http://services.baxus.co/api/bar/user/{username}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # Returns user's bottle collection data
    else:
        logging.error(f"Error fetching BAXUS data for {username}: {response.status_code} - {response.text}")
        raise HTTPException(status_code=500, detail="Error fetching BAXUS data")

# Function to make recommendations using GPT (with file ID)
def get_gpt_recommendations(user_data: dict, db: Session):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"Based on the following user data, recommend personalized whisky bottles from this list of 501 available bottles: {user_data}"

    # Fetch only 501 available bottles from the database
    query = select(Bottle).limit(501)  # Ensure we limit to 501 bottles
    result = db.execute(query)
    available_bottles = result.scalars().all()

    # Prepare the list of available bottles
    available_bottles_data = [{"id": bottle.id, "name": bottle.name, "spirit_type": bottle.spirit_type, "fair_price": bottle.fair_price, "abv": bottle.abv, "ranking": bottle.ranking} for bottle in available_bottles]

    # Add available bottles to the prompt
    data = {
        "model": "gpt-4",  # Use GPT-4
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who provides whisky recommendations."},
            {"role": "user", "content": f"User data: {user_data}\n\nAvailable bottles: {available_bottles_data}\n\nFile ID: {MCP_FILE_ID}"}
        ],
        "max_tokens": 150
    }

    # Call the GPT API to get recommendations (using file ID)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        error_details = response.json()
        logging.error(f"GPT API Error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error with GPT API: {error_details}")

# FastAPI Endpoints

# Endpoint for rating a bottle (like/dislike)
@app.post("/rate_bottle")
async def rate_bottle(rating: RatingModel, db: Session = Depends(get_db)):
    # Save user rating in the database using username
    db_rating = UserRating(user_id=rating.username, bottle_id=rating.bottle_id, liked=rating.liked)  # Using username
    db.add(db_rating)
    db.commit()
    return {"message": "Rating recorded"}

# Endpoint to fetch recommendations based on user ratings
@app.post("/recommendations")
async def recommendations(username: str, db: Session = Depends(get_db)):  # Using username
    try:
        # Fetch user's bar from BAXUS API
        user_data_from_baxus = get_user_bar_data(username)
        
        # Extract bottle data from the BAXUS response
        bottles = [entry["product"] for entry in user_data_from_baxus if "product" in entry]

        recommendation_data = {"username": username, "bar": bottles}  # Using username

        # Fetch user ratings from the database to refine recommendations
        query = select(UserRating).filter(UserRating.user_id == username)  # Using username
        result = db.execute(query)
        user_ratings = result.scalars().all()

        if user_ratings:
            user_data = {"username": username, "ratings": [{"bottle_id": rating.bottle_id, "liked": rating.liked} for rating in user_ratings]}  # Using username
            recommendation_data["ratings"] = user_data["ratings"]

        # Get recommendations based on user's data and ratings (using MCP file ID)
        recommendations = get_gpt_recommendations(recommendation_data, db)
        
        return {"recommendations": recommendations}
    
    except Exception as e:
        logging.error(f"Error fetching recommendations for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching recommendations")

# To run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
