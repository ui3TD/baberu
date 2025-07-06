from os import environ
import logging

from .llm import base as llm_base, gemini, claude, grok, openai, deepseek, openrouter
from .image import base as image_base, imagen, dalle
from .transcription import base as transcription_base, elevenlabs

class AIToolFactory:
    @staticmethod
    def get_llm_provider(model_name: str, system_prompt: str = "") -> llm_base.LLMProvider:
        logger = logging.getLogger(__name__)

        if "/" in model_name.lower():
            api_key = environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                logger.error("OpenRouter API key not found.")
                raise ValueError("OpenRouter API key not found.")
            return openrouter.OpenRouterProvider(api_key=api_key, model=model_name, system_prompt=system_prompt)
        elif 'gemini' in model_name.lower():
            api_key = environ.get("GEMINI_API_KEY", "")
            if not api_key:
                logger.error("Google API key not found.")
                raise ValueError("Google API key not found.")
            return gemini.GeminiProvider(api_key=api_key, model=model_name, system_prompt=system_prompt)
        elif "claude" in model_name.lower():
            api_key = environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warnerroring("Anthropic API key not found.")
                raise ValueError("Anthropic API key not found.")
            return claude.ClaudeProvider(api_key=api_key, model=model_name, system_prompt=system_prompt)
        elif "grok" in model_name.lower():
            api_key = environ.get("XAI_API_KEY", "")
            if not api_key:
                logger.error("XAI API key not found.")
                raise ValueError("XAI API key not found.")
            return grok.GrokProvider(api_key=api_key, model=model_name, system_prompt=system_prompt)
        elif "o1" in model_name.lower() or "o3" in model_name.lower() or "o4" in model_name.lower():
            api_key = environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("OpenAI API key not found.")
                raise ValueError("OpenAI API key not found.")
            return openai.OProvider(api_key=api_key, model=model_name, system_prompt=system_prompt)
        elif "gpt" in model_name.lower():
            api_key = environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("OpenAI API key not found.")
                raise ValueError("OpenAI API key not found.")
            return openai.GPTProvider(api_key=api_key, model=model_name, system_prompt=system_prompt)
        elif "deepseek" in model_name.lower():
            api_key = environ.get("DEEP_API_KEY", "")
            if not api_key:
                logger.error("Deepseek API key not found.")
                raise ValueError("Deepseek API key not found.")
            return deepseek.DeepseekProvider(api_key=api_key, model=model_name, system_prompt=system_prompt)
        else:
            raise ValueError(f"Could not determine LLM provider for model: {model_name}")
        
    @staticmethod
    def get_image_provider(model_name: str) -> image_base.ImageProvider:
        logger = logging.getLogger(__name__)

        if 'imagen' in model_name.lower():
            api_key = environ.get("GEMINI_API_KEY", "")
            if not api_key:
                logger.error("Google API key not found.")
                raise ValueError("Google API key not found.")

            return imagen.ImagenProvider(api_key=api_key, model=model_name)
        elif 'dall-e' in model_name.lower():
            api_key = environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("OpenAI API key not found.")
                raise ValueError("OpenAI API key not found.")
            return dalle.DallEProvider(api_key=api_key, model=model_name)
        else:
            raise ValueError(f"Could not determine image generation provider for model: {model_name}")

    @staticmethod
    def get_transcription_provider(model_name: str) -> transcription_base.TranscriptionProvider:
        logger = logging.getLogger(__name__)

        if 'scribe' in model_name.lower():
            api_key = environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                logger.error("ElevenLabs API key not found.")
                raise ValueError("OpenAI API key not found.")
            return elevenlabs.ScribeProvider(api_key=api_key, model=model_name)
        else:
            raise ValueError(f"Could not determine transcription provider for model: {model_name}")