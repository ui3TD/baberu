from .base import LLMProvider
from google.genai import Client
from google.genai.types import Content, Part, SafetySetting, Tool, GoogleSearch, GenerateContentConfig
import json

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str = None):
        super().__init__(api_key, model, system_prompt)
        self.client = Client(api_key=self.api_key)

    def prompt(self, user_prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Sends a prompt to Gemini and returns the response.
        
        Supported kwargs:
        - grounding (bool): Enable Google Search grounding
        """
        if system_prompt is None:
            system_prompt = self.system_prompt
            
        grounding = kwargs.get("grounding", False)

        prompt_messages = [
            Content(parts=[Part(text=user_prompt)], role="user")
            ]
        
        safety_settings = [
            SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE'
            ),
            SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
            ),
            SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold='BLOCK_NONE'
            ),
            SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE'
            )
        ]
        
        config_params = {
            'system_instruction': system_prompt,
            'safety_settings': safety_settings
        }
        
        if grounding:
            config_params['tools'] = [Tool(google_search=GoogleSearch())]
        
        self.logger.debug(f"Gemini prompt messages:\n{prompt_messages}")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt_messages,
            config=GenerateContentConfig(**config_params)
        )
        self.logger.debug(f"Gemini response:\n{json.dumps(response, indent=2)}")

        if not hasattr(response, 'text') or not response.text:
            self.logger.warning("Gemini returned empty or invalid response")
            if hasattr(response, 'model_dump_json'):
                self.logger.warning(f"Response details: {response.model_dump_json()}")
            return ""

        return response.text