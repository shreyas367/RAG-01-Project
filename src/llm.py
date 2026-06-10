import requests


def generate_answer(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    print("Status Code:", response.status_code)
    print("Response:", response.text)

    response.raise_for_status()

    return response.json()["response"]