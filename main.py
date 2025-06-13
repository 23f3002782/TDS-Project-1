import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients (only what's needed for querying)
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone.Index("tds-embeddings-hf")  # Use your existing index

class LightweightHuggingFaceLLM:
    """Lightweight version for production - only generation and embedding for queries"""
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
        
        # Initialize Hugging Face client for text generation
        self.client = InferenceClient(
            provider="featherless-ai",
            api_key=self.hf_token,
        )
        
        # Only load embedding model for user queries (much smaller memory footprint)
        try:
            self.embedding_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", 
                trust_remote_code=True
            )
            logger.info("Query embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 800) -> str:
        try:
            completion = self.client.chat.completions.create(
                model="mistralai/Magistral-Small-2506",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise Exception(f"Generation failed: {str(e)}")
        
    def generate_completion_with_image(self, messages: List[Dict[str, Any]], temperature: float = 0.7, max_tokens: int = 800) -> str:
        """Handle image + text - extract text and note image limitation for non-vision models"""
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
                        # For non-vision models, we can't process the image
                        text_parts.append("[IMAGE PROVIDED - Current model cannot analyze images. Please describe the image or use a vision-capable model.]")
                
                text_messages.append({
                    "role": msg["role"],
                    "content": " ".join(text_parts)
                })
            else:
                text_messages.append(msg)
        
        # Add note about image limitation if image was provided
        if has_image:
            text_messages.insert(-1, {
                "role": "system",
                "content": "Note: An image was provided but the current model cannot analyze images. Focus on the text content and ask the user to describe the image if visual analysis is needed."
            })
        
        return self.generate_completion(text_messages, temperature, max_tokens)
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise Exception(f"Failed to get embedding: {str(e)}")

def process_image(base64_image: str) -> str:
    """Process and validate base64 image"""
    try:
        # Handle data URL format
        if base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        
        # Validate it's a proper image
        img = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert back to base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        processed_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return processed_base64
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise ValueError(f"Invalid image format: {str(e)}")

# Initialize lightweight LLM
hf_llm = LightweightHuggingFaceLLM()

# Your existing API models and functions remain the same
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkResponse(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkResponse] = []

# Keep your existing semantic_search and generate_answer_with_links functions
# (They only query the existing Pinecone index, no heavy processing)

app = FastAPI(title="TDS Virtual TA", description="Virtual Teaching Assistant")

def semantic_search(query: str, top_k: int = 5, source_filter: str = None) -> List[Dict[str, Any]]:
    """Search for relevant content using embeddings"""
    try:
        query_embedding = hf_llm.get_embedding(query)
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

def generate_answer_with_links(query: str, context_results: List[Dict[str, Any]], image_base64: Optional[str] = None) -> Dict[str, Any]:
    """Generate answer using context and optionally process image"""
    # Separate context by source type
    discourse_context = []
    markdown_context = []
    links = []
    
    for result in context_results:
        try:
            if result["source_type"] == "discourse":
                discourse_context.append(f"Forum Discussion - {result['topic_title']}:\n{result['content']}")
                if result.get("url"):
                    content_preview = result['content'][:200].strip()
                    if len(content_preview) > 150:
                        content_preview = content_preview[:150] + "..."
                    
                    links.append({
                        "url": result["url"],
                        "text": f"Forum: {result['topic_title']} - {content_preview}"
                    })
            else:
                markdown_context.append(f"Course Material - {result['title']}:\n{result['content']}")
        except Exception as e:
            logger.error(f"Error processing result: {e}")
            continue
    
    # Combine contexts
    context_parts = []
    if discourse_context:
        context_parts.append("=== FORUM DISCUSSIONS ===\n\n" + "\n\n---\n\n".join(discourse_context))
    if markdown_context:
        context_parts.append("=== COURSE MATERIALS ===\n\n" + "\n\n---\n\n".join(markdown_context))
    
    if not context_parts:
        context = "No relevant context found."
    else:
        context = "\n\n" + "\n\n".join(context_parts)
    
    # Prepare messages for completion
    try:
        if image_base64:
            # Process and validate the image first
            processed_image = process_image(image_base64)
            # Create proper data URL format for the image
            image_data_url = f"data:image/jpeg;base64,{processed_image}"
            
            messages = [
                {"role": "system", "content": "You are a helpful teaching assistant that answers questions based on forum discussions, course materials, and images. When answering, clearly indicate whether your information comes from forum discussions, course materials, or image analysis. Be concise but thorough."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Based on the following content:\n\n{context}\n\nQuestion: {query}\n\nPlease also analyze the provided image and incorporate relevant information from it in your answer."},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]}
            ]
            
            answer = hf_llm.generate_completion_with_image(messages, max_tokens=800)
        else:
            # Handle text-only query
            messages = [
                {"role": "system", "content": "You are a helpful teaching assistant that answers questions based on forum discussions and course materials. When answering, clearly indicate whether your information comes from forum discussions, course materials, or both. Be concise but thorough."},
                {"role": "user", "content": f"Based on the following content:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            
            answer = hf_llm.generate_completion(messages, max_tokens=800)
        
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


@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint - now only queries existing data"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Query existing Pinecone index
        results = semantic_search(request.question, top_k=5)
        
        # Generate answer using existing data
        response_data = generate_answer_with_links(
            request.question, 
            results, 
            request.image
        )
        
        return AnswerResponse(
            answer=response_data["answer"],
            links=[LinkResponse(**link) for link in response_data["links"]]
        )
        
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Using existing Pinecone database"}

@app.get("/")
async def root():
    return {"message": "TDS Virtual TA API", "version": "1.0 (Production)"}

if __name__ == "__main__":
    # No data initialization needed!
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        workers=1
    )