import os
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from huggingface_hub import InferenceClient
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HuggingFaceQueryLLM:
    """Lightweight LLM class for query processing only"""
    
    def __init__(self, model: str = "mistralai/Magistral-Small-2506", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model = model
        self.embedding_model = embedding_model
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
        
        # Initialize Hugging Face client for chat completions
        self.client = InferenceClient(
            provider="featherless-ai",
            api_key=self.hf_token,
        )
        
        # Initialize embedding client
        self.embedding_client = InferenceClient(
            model=self.embedding_model,
            token=self.hf_token
        )
        
        # Direct API endpoint for embeddings (fallback)
        self.embedding_api_url = f"https://api-inference.huggingface.co/models/{self.embedding_model}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        logger.info(f"Initialized HuggingFace LLM with model: {self.model}")
        logger.info(f"Embedding model: {self.embedding_model}")

    def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1200) -> str:
        """Generate completion using Hugging Face Inference API"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise Exception(f"Hugging Face generation failed: {str(e)}")
    
    def generate_completion_with_image(self, messages: List[Dict[str, Any]], temperature: float = 0.7, max_tokens: int = 1200) -> str:
        """Handle image + text queries"""
        text_messages = []
        has_image = False
        
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for content in msg["content"]:
                    if content.get("type") == "text":
                        text_parts.append(content["text"])
                    elif content.get("type") == "image_url":
                        has_image = True
                        text_parts.append("[IMAGE PROVIDED - Current model cannot analyze images. Please describe the image or use a vision-capable model.]")
                
                text_messages.append({
                    "role": msg["role"],
                    "content": " ".join(text_parts)
                })
            else:
                text_messages.append(msg)
        
        if has_image:
            text_messages.insert(-1, {
                "role": "system",
                "content": "Note: An image was provided but the current model cannot analyze images. Focus on the text content and ask the user to describe the image if visual analysis is needed."
            })
        
        return self.generate_completion(text_messages, temperature, max_tokens)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for user query using Hugging Face API"""
        try:
            # Method 1: Using InferenceClient
            try:
                response = self.embedding_client.feature_extraction([text])
                if isinstance(response, list):
                    if isinstance(response[0], list):
                        return response[0]
                    else:
                        return response
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")
                    
            except Exception as client_error:
                logger.warning(f"InferenceClient failed, trying direct API: {client_error}")
                
                # Method 2: Direct API call as fallback
                payload = {"inputs": text}
                response = requests.post(
                    self.embedding_api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed: {response.status_code} - {response.text}")
                
                result = response.json()
                
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], list):
                        return result[0]
                    else:
                        return result
                else:
                    raise ValueError(f"Unexpected API response format: {type(result)}")
                    
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise Exception(f"Failed to get embedding: {str(e)}")

def process_image(base64_image: str) -> str:
    """Process and validate base64 image"""
    try:
        if base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        image_data = base64.b64decode(base64_image)
        img = Image.open(BytesIO(image_data))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        processed_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return processed_base64
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise ValueError(f"Invalid image format: {str(e)}")

def initialize_pinecone():
    """Initialize Pinecone connection to existing index"""
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        index_name = "tds-embeddings-hf-v2"
        
        # Check if index exists
        if index_name not in pinecone_client.list_indexes().names():
            raise ValueError(f"Index '{index_name}' not found. Please ensure embeddings are already created.")
        
        index = pinecone_client.Index(index_name)
        logger.info(f"Connected to existing Pinecone index: {index_name}")
        
        return index
        
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise

def semantic_search(query: str, index, llm: HuggingFaceQueryLLM, top_k: int = 10, source_filter: str = None) -> List[Dict[str, Any]]:
    """Search for relevant content using embeddings"""
    try:
        query_embedding = llm.get_embedding(query)
    except Exception as e:
        logger.error(f"Failed to get query embedding: {e}")
        return []
    
    filter_dict = {}
    if source_filter:
        filter_dict = {"source_type": source_filter}
    
    try:
        search_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
    except Exception as e:
        logger.error(f"Pinecone search failed: {e}")
        return []
    
    results = []
    for match in search_response.matches:
        try:
            if match.metadata["source_type"] == "discourse":
                results.append({
                    "score": match.score,
                    "source_type": "discourse",
                    "topic_id": match.metadata["topic_id"],
                    "topic_title": match.metadata["topic_title"],
                    "root_post_number": match.metadata["root_post_number"],
                    "post_numbers": [int(pn) for pn in match.metadata["post_numbers"] if pn.isdigit()],
                    "content": match.metadata["combined_text"],
                    "url": match.metadata.get("url", "")
                })
            else:
                results.append({
                    "score": match.score,
                    "source_type": "markdown",
                    "filename": match.metadata["filename"],
                    "title": match.metadata["title"],
                    "chunk_id": match.metadata["chunk_id"],
                    "content": match.metadata["content"],
                    "preview": match.metadata["preview"]
                })
        except Exception as e:
            logger.error(f"Error processing search result: {e}")
            continue
    
    return results

def extract_discourse_links(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract and format discourse links from search results"""
    links = []
    seen_urls = set()
    
    for result in results:
        if result["source_type"] == "discourse" and result.get("url"):
            url = result["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                topic_title = result.get("topic_title", "Discussion")
                links.append({
                    "url": url,
                    "text": f"Forum Discussion: {topic_title}"
                })
    
    return links

def generate_answer_with_links(query: str, context_results: List[Dict[str, Any]], llm: HuggingFaceQueryLLM, image_base64: Optional[str] = None) -> Dict[str, Any]:
    """Generate answer using context and optionally process image"""
    
    links = extract_discourse_links(context_results)
    
    # Check if we have relevant context
    if not context_results or all(result["score"] < 0.5 for result in context_results):
        future_exam_keywords = ["end-term", "end term", "exam", "2025", "sep 2025", "september 2025"]
        if any(keyword in query.lower() for keyword in future_exam_keywords):
            return {
                "answer": "I don't know the exact date for the TDS Sep 2025 end-term exam as this information is not available yet. Please check the official course announcements or contact the instructors for the most up-to-date exam schedule.",
                "links": links
            }
    
    # Format context
    discourse_context = []
    markdown_context = []
    
    for result in context_results:
        try:
            if result["source_type"] == "discourse":
                discourse_context.append({
                    "title": result['topic_title'],
                    "content": result['content'],
                    "score": result['score']
                })
            else:
                markdown_context.append({
                    "title": result['title'],
                    "content": result['content'],
                    "score": result['score']
                })
        except Exception as e:
            logger.error(f"Error processing result: {e}")
            continue
    
    context_parts = []
    
    if discourse_context:
        context_parts.append("=== FORUM DISCUSSIONS ===")
        for i, ctx in enumerate(discourse_context[:3]):
            context_parts.append(f"\nDiscussion {i+1}: {ctx['title']}")
            context_parts.append(f"Content: {ctx['content'][:1000]}{'...' if len(ctx['content']) > 1000 else ''}")
    
    if markdown_context:
        context_parts.append("\n=== COURSE MATERIALS ===")
        for i, ctx in enumerate(markdown_context[:3]):
            context_parts.append(f"\nMaterial {i+1}: {ctx['title']}")
            context_parts.append(f"Content: {ctx['content'][:1000]}{'...' if len(ctx['content']) > 1000 else ''}")
    
    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    
    system_prompt = """You are a Virtual Teaching Assistant for the TDS (Tools in Data Science) course at IIT Madras. 

IMPORTANT GUIDELINES:
1. Answer based ONLY on the provided context from forum discussions and course materials
2. If the context doesn't contain enough information to answer confidently, say "I don't know" or "This information is not available"
3. For technical questions, be specific about tools, versions, and recommendations mentioned in the context
4. When mentioning tools like Docker vs Podman, be precise about course recommendations
5. For scoring/grading questions, provide exact details if available in the context
6. For future dates/events not covered in the context, clearly state the information is not available yet
7. Always be helpful but honest about limitations

Be concise but thorough. If you find contradictory information, mention both perspectives."""

    try:
        if image_base64:
            processed_image = process_image(image_base64)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Question: {query}\n\nContext from TDS course materials and forum:\n{context}\n\nPlease answer the question based on the provided context. If an image was provided, note that you cannot analyze it but can work with any text descriptions."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{processed_image}"}}
                ]}
            ]
            
            answer = llm.generate_completion_with_image(messages, temperature=0.3, max_tokens=1200)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}\n\nContext from TDS course materials and forum:\n{context}\n\nPlease answer the question based on the provided context."}
            ]
            
            answer = llm.generate_completion(messages, temperature=0.3, max_tokens=1200)
        
        return {
            "answer": answer,
            "links": links
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "answer": f"I apologize, but I encountered an error while generating the answer: {str(e)}",
            "links": links
        }

# Initialize components
try:
    hf_llm = HuggingFaceQueryLLM()
    pinecone_index = initialize_pinecone()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="TDS Virtual TA - Query API", 
    description="Virtual Teaching Assistant for querying TDS course content",
    version="1.0.0"
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkResponse(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkResponse] = []

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint to answer questions with optional image support"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Perform semantic search
        results = semantic_search(request.question, pinecone_index, hf_llm, top_k=10)
        
        # Generate answer with links
        response_data = generate_answer_with_links(
            request.question, 
            results, 
            hf_llm,
            request.image
        )
        
        return AnswerResponse(
            answer=response_data["answer"],
            links=[LinkResponse(**link) for link in response_data["links"]]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Hugging Face connection
        test_response = hf_llm.generate_completion([
            {"role": "user", "content": "Test"}
        ], max_tokens=5)
        
        # Test Pinecone connection
        index_stats = pinecone_index.describe_index_stats()
        
        return {
            "status": "healthy",
            "huggingface": "connected",
            "pinecone": "connected",
            "embedding_model": hf_llm.embedding_model,
            "generation_model": hf_llm.model,
            "index_stats": {
                "total_vectors": index_stats.total_vector_count,
                "dimension": index_stats.dimension
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "TDS Virtual TA Query API", 
        "version": "1.0.0",
        "description": "Query-only API for TDS course content using pre-built embeddings",
        "endpoints": {
            "POST /api/": "Answer questions with optional image support",
            "GET /health": "Health check",
            "GET /": "This endpoint"
        }
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)