import os
from openai import OpenAI
import yaml

def get_openai_client():
    # Try to get API key from environment variable first
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # If not found in environment, try to get from config file
    if not api_key:
        try:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            api_key = config['api_credentials']['openai']['api_key']
        except (FileNotFoundError, KeyError):
            raise ValueError("OpenAI API key not found in environment or config file")
    
    return OpenAI(api_key=api_key)

def get_completion_and_token_count(client,prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=100): 
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Extracting the content
        content = response.choices[0].message.content

        # Extracting token usage
        token_dict = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }

        return content, token_dict

    except Exception as e:
        print(f"Error during API call: {e}")
        return None, {}