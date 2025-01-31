import json
import os
from fastapi import FastAPI, Depends, HTTPException, Header, Request, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, ConfigDict
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Annotated, Callable, Union
from pydantic_core import core_schema
import re
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import google.generativeai as genai
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://guardmind-backend.onrender.com", "https://guard-mind.vercel.app"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#MongoDB
uri = os.environ.get("MONGODB_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["mental_health_app"]
users_collection = db["users"]
journals_collection = db["journals"]
community_posts_collection = db["community_posts"]
chats_collection = db["chats"]

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Security
SECRET_KEY = os.environ.get("SECRET_KEY") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ObjectId handling for Pydantic
class _ObjectIdPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        def validate_from_str(input_value: str) -> ObjectId:
            return ObjectId(input_value)

        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(ObjectId),
                core_schema.no_info_plain_validator_function(validate_from_str),
            ],
            serialization=core_schema.to_string_ser_schema(),
        )

PydanticObjectId = Annotated[ObjectId, _ObjectIdPydanticAnnotation]

# Models
class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[PydanticObjectId] = Field(None, alias='_id')
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class JournalEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[PydanticObjectId] = Field(None, alias='_id')
    content: str
    date: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None

class PostCreate(BaseModel):
    content: str

class ReplyCreate(BaseModel):
    content: str


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, handler):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema: Any) -> None:
        field_schema.update(type="string")

class Reply(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    content: str
    username: str
    date: datetime
    user_id: str

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class Post(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    content: str
    username: str
    date: datetime
    user_id: str
    is_owner: bool
    replies: Optional[List[Reply]] = []

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
class ChatMessage(BaseModel):
    sender: str
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)

class Chat(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[PydanticObjectId] = Field(None, alias='_id')
    user_id: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    user = users_collection.find_one({"username": username})
    if not user or not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Add debug logging
    print(f"Authenticating token: {token[:10]}..." if token and len(token) > 10 else token)
    
    if token == "guest":
        return {"username": "guest", "id": "guest", "is_guest": True}
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify and decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if not username:
            print("No username found in token payload")
            raise credentials_exception
            
        # Find user in database
        user = users_collection.find_one({"username": username})
        
        if not user:
            print(f"No user found for username: {username}")
            raise credentials_exception
            
        # Convert ObjectId to string for JSON serialization
        user["_id"] = str(user["_id"])
        
        # Add last login time if you want to track it
        users_collection.update_one(
            {"_id": ObjectId(user["_id"])},
            {"$set": {"last_login": datetime.now()}}
        )
        
        return user
        
    except JWTError as e:
        print(f"JWT decode error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        print(f"Unexpected authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )
# Routes
@app.get("/health")
async def health_check():
    return {"status": "ok"}
    
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    db_user = users_collection.find_one({"username": form_data.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user_id": str(db_user["_id"])}

@app.post("/register")
async def register(user: User):
    existing_user = users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    user_dict = user.model_dump(exclude={'id'})
    user_dict["password"] = hashed_password
    result = users_collection.insert_one(user_dict)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/journal", response_model=Union[List[JournalEntry], Dict[str, str]])
async def get_journal_entries(current_user: dict = Depends(get_current_user)):
    if current_user.get("is_guest", False):
        return {"message": "Please login to view your Journal Entries"}
    
    entries = list(journals_collection.find({"user_id": str(current_user["_id"])}))
    return [JournalEntry(**entry) for entry in entries]

@app.post("/journal", response_model=Dict[str, str])
async def create_journal_entry(entry: JournalEntry, current_user: dict = Depends(get_current_user)):
    if current_user.get("is_guest", False):
        raise HTTPException(status_code=403, detail="Please login to create Journal Entries")
    
    try:
        entry_dict = entry.model_dump(exclude={'id'})
        entry_dict["user_id"] = str(current_user["_id"])
        result = journals_collection.insert_one(entry_dict)
        return {"message": "Journal entry created", "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error creating journal entry: {str(e)}")

@app.delete("/journal/{entry_id}", response_model=Dict[str, str])
async def delete_journal_entry(entry_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("is_guest", False):
        raise HTTPException(status_code=403, detail="Please login to delete Journal Entries")
    
    result = journals_collection.delete_one({"_id": ObjectId(entry_id), "user_id": str(current_user["_id"])})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    
    return {"message": "Journal entry deleted successfully"}

# Community Router
router = APIRouter()
# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY"),
#     base_url= os.environ.get("OPENAI_BASE_URL")
# )
# #Chatbot Logic
# def ai_response(text):
#     completion = client.chat.completions.create(
#         model="pai-001",
#         messages=[
#             {"role": "user", "content": "You are a mental health professional. You are a helpful therapy assistant bot. Please respond to the following message from the patient. Do not respond more than 2 sentences. And most importantly, just answer the patient's question and dont ask another question as a response to the question. The patient says: " + text},
#         ],
#     )

#     return completion.choices[0].message.content
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ai_response(text):
    try:
        prompt = "You are a mental health professional. You are a helpful therapy assistant bot. Please respond to the following message from the patient. Make sure you are helpful in solving their issue. Do not answer more than 100 words. And most importantly, just answer the patient's question and dont ask another question as a response to the question. The patient says: " + text
        
        # Generate content using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        response = model.generate_content(prompt)
        
        return response.text.strip()
    except Exception as e:
        print(f"Error in AI response: {e}")
        return "I apologize, but I'm unable to provide a response at the moment."

@app.post("/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    is_guest = current_user.get("is_guest", False)
    
    if is_guest:
        user_id = str(current_user["id"])
        try:
            response = ai_response(request.message)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating response")
        
        return {"response": response, "chat_id": None}
    user_id = str(current_user["_id"])
    if request.chat_id:
        chat = chats_collection.find_one({"_id": ObjectId(request.chat_id), "user_id": user_id})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
    else:
        chat = {"user_id": user_id, "messages": [], "created_at": datetime.now()}
    
    chat["messages"].append({"sender": "user", "text": request.message, "timestamp": datetime.now()})
    
    try:
        response = ai_response(request.message)
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response")
    
    chat["messages"].append({"sender": "bot", "text": response, "timestamp": datetime.now()})

    if request.chat_id:
        chats_collection.update_one({"_id": ObjectId(request.chat_id)}, {"$set": chat})
    else:
        result = chats_collection.insert_one(chat)
        request.chat_id = str(result.inserted_id)
    
    return {"response": response, "chat_id": request.chat_id}


@app.get("/chat-history")
async def get_chat_history(current_user: dict = Depends(get_current_user)):
    if current_user.get("is_guest", False):
        return []
    
    user_id = str(current_user["_id"])
    chats = list(chats_collection.find({"user_id": user_id}).sort("created_at", -1))
    return [{"id": str(chat["_id"]), "created_at": chat["created_at"]} for chat in chats]


@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("is_guest", False):
        raise HTTPException(status_code=403, detail="Guest users cannot access chat history")
    
    user_id = str(current_user["_id"])
    chat = chats_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"messages": chat["messages"]}

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("is_guest", False):
        raise HTTPException(status_code=403, detail="Guest users cannot delete chats")
    
    user_id = str(current_user["_id"])
    result = chats_collection.delete_one({"_id": ObjectId(chat_id), "user_id": user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found or you don't have permission to delete it")
    
    return {"message": "Chat deleted successfully"}
@app.post("/community", response_model=Post)
async def create_post(post: PostCreate, current_user: dict = Depends(get_current_user)):
    post_dict = post.model_dump()
    post_dict["user_id"] = str(current_user["_id"])
    post_dict["username"] = current_user["username"]
    post_dict["date"] = datetime.now()
    result = community_posts_collection.insert_one(post_dict)
    created_post = community_posts_collection.find_one({"_id": result.inserted_id})
    return Post(
        **created_post,
        is_owner=True
    )

@app.get("/community", response_model=Union[List[Post], Dict[str, str]])
async def get_posts(current_user: dict = Depends(get_current_user)):
    if current_user.get("is_guest", False):
        return {"message": "Please login to view the Community Posts"}
    
    posts = list(community_posts_collection.find().sort("date", -1))
    return [
        Post(
            id=post["_id"],
            content=post["content"],
            username=post["username"],
            date=post["date"],
            user_id=post["user_id"],
            is_owner=str(post["user_id"]) == str(current_user["_id"]),
            replies=[
                Reply(
                    id=reply["_id"],
                    content=reply["content"],
                    username=reply["username"],
                    date=reply["date"],
                    user_id=reply["user_id"]
                ) for reply in post.get("replies", [])
            ]
        ) for post in posts
    ]

@app.post("/community/{post_id}/reply", response_model=Post)
async def create_reply(post_id: str, reply: ReplyCreate, current_user: dict = Depends(get_current_user)):
    try:
        reply_dict = reply.model_dump()
        reply_dict["user_id"] = str(current_user["_id"])
        reply_dict["username"] = current_user["username"]
        reply_dict["date"] = datetime.now()
        reply_dict["_id"] = ObjectId()
        
        result = community_posts_collection.update_one(
            {"_id": ObjectId(post_id)},
            {"$push": {"replies": reply_dict}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Post not found")
        
        updated_post = community_posts_collection.find_one({"_id": ObjectId(post_id)})
        
        post_data = {k: v for k, v in updated_post.items() if k != 'replies'}
        post = Post(
            **post_data,
            id=str(updated_post["_id"]),
            is_owner=str(updated_post["user_id"]) == str(current_user["_id"])
        )
        
        post.replies = [Reply(**reply) for reply in updated_post.get("replies", [])]
        
        return post
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while creating the reply: {str(e)}")


@app.post("/community/{post_id}/reply", response_model=Post)
async def create_reply(post_id: str, reply: ReplyCreate, current_user: dict = Depends(get_current_user)):
    try:
        reply_dict = reply.model_dump()
        reply_dict["user_id"] = str(current_user["_id"])
        reply_dict["username"] = current_user["username"]
        reply_dict["date"] = datetime.now()
        reply_dict["_id"] = ObjectId()
        
        result = community_posts_collection.update_one(
            {"_id": ObjectId(post_id)},
            {"$push": {"replies": reply_dict}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Post not found")
        
        updated_post = community_posts_collection.find_one({"_id": ObjectId(post_id)})
        return Post(
            **updated_post,
            is_owner=str(updated_post["user_id"]) == str(current_user["_id"]),
            replies=[Reply(**reply) for reply in updated_post.get("replies", [])]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while creating the reply: {str(e)}")

@app.delete("/community/{post_id}", response_model=dict)
async def delete_post(post_id: str, current_user: dict = Depends(get_current_user)):
    try:
        post_object_id = ObjectId(post_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid post ID")

    post = community_posts_collection.find_one({"_id": post_object_id})

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if str(post["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="You don't have permission to delete this post")

    result = community_posts_collection.delete_one({"_id": post_object_id})

    if result.deleted_count == 1:
        return {"message": "Post deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete the post")
    
app.include_router(router, prefix="/api/v1/community")

#PPTM Classes
class PsychologicalMetric(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[PydanticObjectId] = Field(None, alias='_id')
    user_id: str
    emotional_stability: float
    social_engagement: float
    cognitive_flexibility: float
    stress_resilience: float
    date: datetime = Field(default_factory=datetime.now)
    health_score: float
    mood: float
    stress: float
    sleep_hours: float
    linguistic_metrics: dict = Field(default_factory=lambda: {
        "emotional_depth": 0,
        "vocabulary_range": 0,
        "sentence_complexity": 0,
        "topic_coherence": 0,
        "expression_clarity": 0
    })
    themes: List[dict] = Field(default_factory=list) 

class UserInsight(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[PydanticObjectId] = Field(None, alias='_id')
    user_id: str
    date: datetime = Field(default_factory=datetime.now)
    narrative_summary: str
    recommendations: List[dict]
    predictions: dict
    triggers: List[str]
    coping_mechanisms: List[str]

psychological_metrics = db["psychological_metrics"]
user_insights = db["user_insights"]

def analyze_text_with_ai(text: str, analysis_type: str) -> dict:
    print(f"\n=== Starting analysis for {analysis_type} ===")
    print(f"Input text (first 100 chars): {text[:100]}")
    
    if not text.strip():
        print("Empty text, returning defaults")
        return get_default_analysis(analysis_type)

    prompt_map = {
    "sentiment": """Analyze the sentiment and emotions in the provided text. Return ONLY a JSON object with this EXACT structure, no comments or explanations:
{
    "positivity": <integer between 0-100>,
    "emotional_stability": <integer between 0-100>,
    "primary_emotions": [<string>, <string>]
}""",

    "metrics": """Analyze the psychological metrics in the provided text. Return ONLY a JSON object with this EXACT structure, no comments or explanations:
{
    "emotional_stability": <integer between 0-100>,
    "social_engagement": <integer between 0-100>,
    "cognitive_flexibility": <integer between 0-100>,
    "stress_resilience": <integer between 0-100>
}""",

    "recommendations": """Based on the text analysis, return ONLY a VALID JSON array with three personalized recommendations in this structure:
[
    {
        "title": <brief title of first recommendation>,
        "description": <detailed description of first recommendation in less than 25 words>
    },
    {
        "title": <brief title of second recommendation>,
        "description": <detailed description of second recommendation in less than 25 words>
    },
    {
        "title": <brief title of third recommendation>,
        "description": <detailed description of third recommendation in less than 25 words>
    }
]""",

    "ai_insights": """Analyze the provided text and activity patterns to generate AI-driven insights. Return ONLY a JSON array with this EXACT structure, no comments or explanations:
[
    {
        "title": <string>,
        "description": <string>
    },
    {
        "title": <string>,
        "description": <string>
    }
]"""
}
    
    try:
        full_prompt = f"""You are a precise JSON generator. Return ONLY valid JSON with the exact structure requested. No comments, explanations, or additional text.

{prompt_map[analysis_type]}

Text to analyze: {text}"""
        
        print(f"\nSending prompt to Gemini...")
        
        # Generate content using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        response = model.generate_content(full_prompt)
        response_content = response.text.strip()
        
        print(f"\nRaw AI response:\n{response_content}")
        
        # Clean response
        if "```" in response_content:
            print("Detected markdown, cleaning...")
            response_content = response_content.split("```")[1]
            if response_content.startswith("json"):
                response_content = response_content[4:]
            response_content = response_content.strip()

        try:
            # Try to extract JSON using regex
            json_pattern = r'(\{.*\}|\[.*\])'
            match = re.search(json_pattern, response_content, re.DOTALL)
            if match:
                response_content = match.group(0)
        except Exception as extract_error:
            print(f"JSON extraction error: {extract_error}")
        
        # Parse the JSON
        result = json.loads(response_content)
        print(f"\nParsed JSON successfully:\n{json.dumps(result, indent=2)}")
        
        return result
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return get_default_analysis(analysis_type)
        
def get_default_analysis(analysis_type: str) -> dict:
    defaults = {
        "sentiment": {
            "positivity": 50,
            "emotional_stability": 50,
            "primary_emotions": ["neutral"]
        },
        "metrics": {
            "emotional_stability": 50,
            "social_engagement": 50,
            "cognitive_flexibility": 50,
            "stress_resilience": 50
        },
        "recommendations": [
            {"title": "Take a Break", "description": "Remember to take regular breaks during your day"},
            {"title": "Stay Active", "description": "Try to maintain regular physical activity"},
            {"title": "Connect with Others", "description": "Reach out to friends or family for support"}
        ],
        "ai_insights": [
            {"title": "Improve Sleep Habits", "description": "Consider maintaining a consistent sleep schedule for better rest."},
            {"title": "Manage Stress Effectively", "description": "Try mindfulness techniques to reduce stress levels."}
        ]
    }
    return defaults.get(analysis_type, {})

@app.get("/pptm/dashboard/{user_id}")
async def get_dashboard_data(user_id: str, current_user: dict = Depends(get_current_user)):
    if str(current_user["_id"]) != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Get recent journal entries with their dates
    recent_journals = list(journals_collection.find(
        {"user_id": user_id},
        {"content": 1, "date": 1}
    ).sort("date", -1).limit(30))  

    # Analyze each journal entry separately for mood tracking
    mood_data = []
    combined_text = ""
    
    for entry in recent_journals:
        # Analyze sentiment for each entry individually
        entry_sentiment = analyze_text_with_ai(entry["content"], "sentiment")
        mood_data.append({
            "date": entry["date"].isoformat(),
            "mood": entry_sentiment.get("positivity", 0)
        })
        combined_text += entry["content"] + " "

    # Analyze combined text for overall metrics
    metrics_analysis = analyze_text_with_ai(combined_text, "metrics")
    recommendations = analyze_text_with_ai(combined_text, "recommendations")

    # Calculate health score
    health_score = (
        metrics_analysis.get("emotional_stability", 0) +
        metrics_analysis.get("social_engagement", 0) +
        metrics_analysis.get("cognitive_flexibility", 0) +
        metrics_analysis.get("stress_resilience", 0)
    ) / 4

    ai_insights = analyze_text_with_ai(combined_text, "ai_insights")

    # Store current metrics
    metric = PsychologicalMetric(
        user_id=user_id,
        emotional_stability=metrics_analysis.get("emotional_stability", 0),
        social_engagement=metrics_analysis.get("social_engagement", 0),
        cognitive_flexibility=metrics_analysis.get("cognitive_flexibility", 0),
        stress_resilience=metrics_analysis.get("stress_resilience", 0),
        health_score=health_score,
        mood=sum(d["mood"] for d in mood_data) / len(mood_data) if mood_data else 0,
        stress=100 - metrics_analysis.get("stress_resilience", 0),
        sleep_hours=8.0
    )
    psychological_metrics.insert_one(metric.model_dump(exclude={'id'}))

    return {
        "current_metrics": {
            "health_score": health_score,
            "psychological_dimensions": metrics_analysis,
            "sentiment": mood_data[-1] if mood_data else {"mood": 0}  
        },
        "historical_data": sorted(mood_data, key=lambda x: x["date"]),  
        "recommendations": recommendations,
          "insights": {
            "triggers": ["Work stress", "Social isolation", "Sleep irregularity"],
            "coping_mechanisms": ["Journaling", "Exercise", "Meditation"]
        },
        "ai_insights": ai_insights
    }
@app.get("/pptm/correlations/{user_id}")
async def get_correlations(user_id: str, current_user: dict = Depends(get_current_user)):
    if str(current_user["_id"]) != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    metrics = list(psychological_metrics.find({"user_id": user_id}))
    correlations = {
        "sleep_mood": calculate_correlation([m["sleep_hours"] for m in metrics], [m["mood"] for m in metrics]),
        "stress_mood": calculate_correlation([m["stress"] for m in metrics], [m["mood"] for m in metrics]),
        "social_stress": calculate_correlation([m["social_engagement"] for m in metrics], [m["stress"] for m in metrics])
    }

    return correlations

def calculate_correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = (sum((xi - x_mean) ** 2 for xi in x) * sum((yi - y_mean) ** 2 for yi in y)) ** 0.5
    return numerator / denominator if denominator != 0 else 0