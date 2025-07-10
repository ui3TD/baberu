try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required to use OpenAI models. "
        "Install it with: pip install openai"
    )

from .base import ImageProvider
from pathlib import Path
import base64

class OpenAIProvider(ImageProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=self.api_key)
    
    def prompt(self, prompt: str, file_path: Path, **kwargs) -> None:
        # Build the base parameters
        params = {
            "model": self.model,
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1
        }
        
        # Only add quality if it should be set
        if "dall-e-3" in self.model.lower():
            params["quality"] = "hd"
        elif "gpt" in self.model.lower():
            params["quality"] = "high"
        
        response = self.client.images.generate(**params)
        self.logger.debug(f"Response from API: {response.model_dump_json()}")

        if "dall-e" in self.model.lower():
            response = response.data[0].url
            self._download_image(response, file_path)
        elif "gpt" in self.model.lower():
            image_base64 = response.model_dump()["data"][0]["b64_json"]
            image_bytes = base64.b64decode(image_base64)
            with open(file_path, "wb") as f:
                f.write(image_bytes)
        return
    
    def _download_image(self, url: str, file_name: Path) -> None:
        """Downloads an image from a URL and saves it locally
        Args:
            url: Source URL of the image
            file_name: Destination file name for saving
        """
        self.logger.info(f"Downloading image from {url}")

        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(file_name, "wb") as f:
                f.write(response.content)

            self.logger.info(f"Image downloaded and saved as '{file_name}'")
            return
        except Exception as e:
            self.logger.error(f"Error downloading image: {str(e)}")
            raise
