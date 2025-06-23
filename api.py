from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
from app import generate_keywords, web_search_free_only, generate_blog

app = FastAPI(
    title="Blog Generation API",
    description="API for generating blog posts using AI",
    version="1.0.0",
)

# CORS middleware ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Güvenlik için burada frontend adresini belirtebilirsin, örn: ["http://localhost:9000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BlogRequest(BaseModel):
    user_input: str


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class BlogResponse(BaseModel):
    query: str
    results_count: int
    search_results: List[SearchResult]
    blog_content: str


@app.post("/generate-blog", response_model=BlogResponse)
async def generate_blog_api(request: BlogRequest):
    try:
        logger.info(f"[API] Starting blog generation process...")

        # Generate keywords from user input
        query = generate_keywords(request.user_input)
        if not query:
            raise HTTPException(
                status_code=400, detail="Could not generate search keywords"
            )

        # Web Searech
        results = web_search_free_only(query)
        if not results:
            raise HTTPException(status_code=404, detail="No search results found")

        # Generate blog
        blog_content = generate_blog(results, request.user_input)
        if not blog_content:
            raise HTTPException(
                status_code=500, detail="Failed to generate blog content"
            )

        return BlogResponse(
            query=query,
            results_count=len(results),
            search_results=[SearchResult(**result) for result in results],
            blog_content=blog_content,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating blog: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
