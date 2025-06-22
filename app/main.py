from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test", tags=["test"])
async def test():
    return {"message": "This is a test endpoint"}
