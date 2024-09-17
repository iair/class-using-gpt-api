import os
import yaml
import openai

def get_openai_client(api_key=None):
    if not api_key:
        try:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            api_key = config['api_credentials']['openai']['api_key']
        except (FileNotFoundError, KeyError):
            raise ValueError("OpenAI API key not found in environment or config file")
    return openai.OpenAI(api_key=api_key)
def get_completion_and_token_count(client,prompt, model, temperature=0.7, max_tokens=100): 
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        token_dict = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }
        return content, token_dict
    except Exception as e:
        print(f"Error during API call: {e}")
        return None, {}    
def check_moderation(client, prompt,model="text-moderation-latest"):
    try:
        response = client.moderations.create(input=prompt,model=model)
        print("Moderation response:", response)
        if response.results[0].flagged:
            print("The input was flagged for violating moderation guidelines.")
        else:
            print("The input was not flagged.")
    except Exception as e:
        print(f"Error: {e}")