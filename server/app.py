from server import app

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
