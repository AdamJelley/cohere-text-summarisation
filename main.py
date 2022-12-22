from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from cohere_endpoints.summarisation import generate_response

app = FastAPI()


@app.get("/")
def read_root():
    with open("templates/index.html") as f:
        html = f.read()
    return HTMLResponse(html)


class TextRequest(BaseModel):
    article: str


@app.post("/summarise")
async def summarise_article(article: TextRequest):
    print(article)
    article = article.dict()["article"]
    article = article.replace("_", " ")

    summary = generate_response(article)

    return summary.generations[0].text


@app.get("/summarise/{article}")
def summarise_article(article: str):
    article = article.replace("_", " ")

    summary = generate_response(article)

    return {"Summary": summary.generations[0].text}
