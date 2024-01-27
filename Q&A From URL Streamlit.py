import streamlit as st
import requests
from bs4 import BeautifulSoup
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview import generative_models
import random


st.set_page_config(layout="wide")

def extract_paragraphs_from_url(url):
    # Send an HTTP request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract paragraphs from the HTML
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

        return paragraphs
    else:
        st.error(f"Error: Unable to fetch content from {url}. Status code: {response.status_code}")
        return None

def generate(input):
    model = generative_models.GenerativeModel("gemini-pro-vision")
    # Safety config
    safety_config = {
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE
    }

    responses = model.generate_content(
        f"""{input}""",
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32
        },
        safety_settings=safety_config
    )

    return responses.candidates[0].content.parts[0].text

def main():
    st.title("Question Answer Generator")

    # Input URL
    url = st.text_input("Enter URL:", 'https://lifeintheuktestweb.co.uk/a-long-and-illustrious-history/')

    # Button to generate question-answer pair
    if st.button("Generate Question Answer Pair"):
        extracted_paragraphs = extract_paragraphs_from_url(url)
        if extracted_paragraphs:
            random_paragraph = random.choice(extracted_paragraphs)
            question_answer_pair = generate(f"generate question answer pair: {random_paragraph}")
            st.success("Question Answer Pair Generated!")
            st.text(question_answer_pair)

    # # Button to show the answer
    # if st.button("Show Answer"):
    #     # Use this space to implement logic for showing the answer if needed
    #     st.warning("You can implement the logic to show the answer here.")

if __name__ == '__main__':
    main()
