from .base import ImageProvider
from openai import OpenAI
from pathlib import Path

class OpenAIProvider(ImageProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=self.api_key)
    
    def prompt(self, prompt: str, file_path: Path, **kwargs) -> None:

        quality = "standard"
        if "dall-e-3" in self.model.lower():
            quality = "hd"
        if "gpt" in self.model.lower():
            quality = "high"
            
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size="1024x1024",
            quality=quality,
            n=1
        )
        response = response.data[0].url
        self._download_image(response, file_path)
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
