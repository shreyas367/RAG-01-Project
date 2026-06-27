import requests


def generate_answer(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5vl:3b",
            "prompt": prompt,
            "stream": False
        }
    )


    response.raise_for_status()

    return response.json()["response"]