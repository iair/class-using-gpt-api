import os
import yaml
import openai

def get_openai_client(api_key=None):
        """
        Returns an OpenAI client instance.
        If an API key is provided, it will be used to authenticate the client.
        Otherwise, the function will attempt to load the API key from a `config.yaml` file.
        
        Args:
            api_key (str, optional): The OpenAI API key. Defaults to None.
        
        Returns:
            openai.OpenAI: The OpenAI client instance.
        
        Raises:
            ValueError: If the API key is not found in the environment or config file.
        """
        if not api_key:
            try:
                with open('config.yaml', 'r') as file:
                    config = yaml.safe_load(file)
                api_key = config['api_credentials']['openai']['api_key']
            except (FileNotFoundError, KeyError):
                raise ValueError("OpenAI API key not found in environment or config file")
        return openai.OpenAI(api_key=api_key)
def get_completion_and_token_count(client,prompt, model, temperature=0.7, max_tokens=100): 
        """
        Sends a completion request to the OpenAI API and returns the response content and token count.
        
        Args:
            client (openai.OpenAI): The OpenAI client instance.
            prompt (str): The prompt to send to the API.
            model (str): The model to use for the completion.
            temperature (float, optional): The temperature to use for the completion. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
        
        Returns:
            tuple: A tuple containing the response content and a dictionary with token counts.
                The dictionary contains the following keys:
                    - 'prompt_tokens': The number of tokens in the prompt.
                    - 'completion_tokens': The number of tokens in the completion.
                    - 'total_tokens': The total number of tokens.
                    
        Raises:
            Exception: If an error occurs during the API call.
        """
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
        """
        Checks if the given prompt violates moderation guidelines.
        Args:
            client (openai.OpenAI): The OpenAI client instance.
            prompt (str): The input text to check for moderation.
            model (str, optional): The moderation model to use. Defaults to "text-moderation-latest".
        Returns:
            None
        Prints:
            The moderation response, including whether the input was flagged or not.
            Any error messages that occur during the API call.
        Raises:
            Exception: If an error occurs during the API call.
        """
        try:
            response = client.moderations.create(input=prompt,model=model)
            print("Moderation response:", response)
            if response.results[0].flagged:
                print("The input was flagged for violating moderation guidelines.")
            else:
                print("The input was not flagged.")
        except Exception as e:
            print(f"Error: {e}")