from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import chatbot_response

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chatbot_response(request.session_id, request.user_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)