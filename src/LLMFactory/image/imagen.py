try:
    from google.genai import Client
    from google.genai.types import GenerateImagesConfig, PersonGeneration, SafetyFilterLevel
except ImportError:
    raise ImportError(
        "The 'google-genai' package is required to use Imagen models. "
        "Install it with: pip install google-genai"
    )

from .base import ImageProvider
from PIL import Image
from io import BytesIO
from pathlib import Path

class ImagenProvider(ImageProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = Client(api_key=self.api_key)
    
    def prompt(self, prompt: str, file_path: Path, **kwargs) -> None:

        response = self.client.models.generate_images(
            model=self.model,
            prompt=prompt,
            config=GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
                safety_filter_level=SafetyFilterLevel.BLOCK_LOW_AND_ABOVE,
                person_generation=PersonGeneration.ALLOW_ADULT
            )
        )
        self.logger.info(response)
        image = Image.open(BytesIO( response.generated_images[0].image.image_bytes))
        response = prompt
        image.save(file_path)

        return