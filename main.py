from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from solver import solve_quiz_chain, EMAIL, SECRET
import asyncio

app = FastAPI(title="TDS LLM Quiz Solver",
              description="Solves TDS LLM Analysis Quiz using LLM reasoning + Python tools.",
              version="1.0")

# -----------------------------
# INPUT MODEL
# -----------------------------
class SolveRequest(BaseModel):
    email: str
    secret: str
    url: str  # first quiz URL


# -----------------------------
# ROOT ENDPOINT
# -----------------------------
@app.get("/")
def home():
    return {"message": "LLM Quiz Solver API is running."}


# -----------------------------
# SOLVE ENDPOINT
# -----------------------------
@app.post("/solve")
async def solve_endpoint(req: SolveRequest):
    # 1. Validate email + secret
    if req.email != EMAIL or req.secret != SECRET:
        raise HTTPException(status_code=403,
                            detail="Email or Secret does not match.")

    # 2. Call quiz solver (async)
    try:
        result = await solve_quiz_chain(req.url)
        return {
            "email": req.email,
            "start_url": req.url,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Solver failed: {e}")

