import requests
from dotenv import load_dotenv
import os
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from treys import Card, Evaluator

# Load API keys from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")


def get_homework_answer(image_url):
    """
    Gets a homework answer from an image URL using OpenAI's GPT-4o-mini model.
    
    Args:
        image_url (str): URL of the image containing the homework question
        
    Returns:
        str: The answer from the AI model
    """

    #probability code
    #dependencies: from treys: Card, Evaluator
    hand1 = "Qs"
    hand2 = "As"
    board1 = "Js"
    board2 = "Ts"
    board3 = "Ks"
    h1 = Card.new(hand1)
    h2 = Card.new(hand2)
    b1 = Card.new(board1)
    b2 = Card.new(board2)
    b3 = Card.new(board3)
    board = [b1, b2, b3]  # Board cards
    hand = [h1, h2]  # Hand cards
    evaluator = Evaluator()
    score = evaluator.evaluate(board, hand)  # Evaluate hand strength/rank
    probability = evaluator.get_five_card_rank_percentage(score)  # Probability of winning
    
    client = OpenAI(api_key=openai_api_key)  # Pass API key directly
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Answer this math problem in ONE LINE max NO MARKDOWN with a {probability}% chance you'll intentionally provide a WRONG answer (no explanation if wrong) based off the image."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.7,
            presence_penalty=0.6,
        )
        
        if response.choices:
            return response.choices[0].message.content
        else:
            raise ValueError("No response received from OpenAI")
            
    except Exception as e:
        print(f"Error getting answer: {str(e)}")
        return None

def speak_text(text):
    """
    Converts text to speech using ElevenLabs API and plays it.
    
    Args:
        text (str): Text to convert to speech
    """    
    client = ElevenLabs(api_key=elevenlabs_api_key)

    # Convert text to speech
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    # Play the audio directly using ElevenLabs play function
    play(audio)

# Example usage:
if __name__ == "__main__":
    image_url = "https://www.mathgptpro.com/static/media/EXAMPLE_1_1_EN.9fda5445065d1bc9bca3.png"
    answer = get_homework_answer(image_url)
    if answer:
        print(f"Answer: {answer}")
        speak_text(answer)
