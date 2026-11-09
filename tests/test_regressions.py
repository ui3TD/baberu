import json
from types import SimpleNamespace

from baberu.providers.llm.openai import GPTProvider
from baberu.providers.transcription.elevenlabs import ScribeProvider

from test_factory_and_parsers import scribe_response


def test_scribe_tolerates_missing_speaker():
    data = scribe_response()
    content = json.loads(data["additional_formats"][0]["content"])
    content["segments"][0]["words"][0]["speaker_id"] = None
    data["additional_formats"][0]["content"] = json.dumps(content)

    result = ScribeProvider.parse(data)

    assert result.segments[0].words[0].speaker == "Default"


def test_gpt_grounding_sends_a_list_of_tools():
    provider = GPTProvider(api_key="k", model="gpt-4.1")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text="ok", model_dump_json=lambda: "{}")

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))

    provider.prompt("hi", grounding=True)

    assert captured["tools"] == [{"type": "web_search_preview"}]
