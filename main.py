from fastapi import FastAPI

from cohere_endpoints.summarisation import generate_response

app = FastAPI()


@app.get("/")
def read_root():
    return (
        "Go to {current_url}/docs and use summarisation endpoint to summarise article."
    )


@app.get("/summarise/{article}")
def summarise_article(article: str):
    article = article.replace("_", " ")

    summary = generate_response(article)

    return {"Summary": summary.generations[0].text}
