"""A video's scenes, and the frames from them that go with translation batches."""
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
from pysubs2 import SSAEvent

from baberu.providers.llm.base import Image
from baberu.tools import av_utils
from baberu.tools.file_utils import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scene:
    """A stretch of video between two cuts, in milliseconds."""
    start_ms: int
    end_ms: int

    @property
    def middle_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2

    @property
    def length_ms(self) -> int:
        return self.end_ms - self.start_ms

    def overlaps(self, event: SSAEvent) -> bool:
        return self.start_ms < max(event.end, event.start + 1) and event.start < self.end_ms


@dataclass(frozen=True)
class Frame:
    """The picture from the middle of a scene."""
    scene: Scene
    image: Image


def scenes_from_cuts(cuts: list[float], duration_sec: float, min_scene_sec: float,
                     max_scene_sec: float) -> list[Scene]:
    """The scenes of a video from its cut times, in seconds, covering it from start to end.

    A scene shorter than `min_scene_sec` joins the scene before it; a short first
    scene joins the one after. A scene longer than `max_scene_sec` is split into
    equal parts no longer than that.
    """
    end_ms = round(duration_sec * 1000)
    if end_ms <= 0:
        return []
    bounds = sorted({0, end_ms, *(round(cut * 1000) for cut in cuts if 0 < cut * 1000 < end_ms)})
    min_ms, max_ms = min_scene_sec * 1000, max_scene_sec * 1000

    spans: list[list[int]] = []
    for start, end in zip(bounds, bounds[1:]):
        if spans and end - start < min_ms:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    if len(spans) > 1 and spans[0][1] - spans[0][0] < min_ms:
        first = spans.pop(0)
        spans[0][0] = first[0]

    scenes: list[Scene] = []
    for start, end in spans:
        parts = max(1, math.ceil((end - start) / max_ms))
        edges = [start + round((end - start) * i / parts) for i in range(parts + 1)]
        scenes.extend(Scene(a, b) for a, b in zip(edges, edges[1:]))
    return scenes


def pick(scenes: list[Scene], events: list[SSAEvent], limit: int) -> list[Scene]:
    """The scenes that overlap at least one event, in time order. Over `limit`, the longest are kept."""
    chosen = [scene for scene in scenes if any(scene.overlaps(event) for event in events)]
    if len(chosen) > limit:
        kept = set(sorted(chosen, key=lambda scene: (-scene.length_ms, scene.start_ms))[:limit])
        chosen = [scene for scene in chosen if scene in kept]
    return chosen


class FrameSource:
    """Frames for subtitle events. Each is grabbed from the video when first asked for and kept in `folder`."""

    def __init__(self, video: Path, scenes: list[Scene], folder: Path, height: int, limit: int):
        self.video = video
        self.scenes = scenes
        self.folder = folder
        self.height = height
        self.limit = limit

    def for_events(self, events: list[SSAEvent]) -> list[Frame]:
        """One frame per scene the events overlap, at most `limit`, in time order. A frame ffmpeg can't grab is left out."""
        frames = []
        for scene in pick(self.scenes, events, self.limit):
            path = self.folder / f"{scene.middle_ms}-{self.height}p.jpg"
            if not path.exists():
                try:
                    av_utils.grab_frame(self.video, scene.middle_ms / 1000, self.height, path)
                except ffmpeg.Error:
                    logger.warning(f"Couldn't grab the frame at {scene.middle_ms / 1000:.1f}s; leaving it out.")
                    continue
            frames.append(Frame(scene, Image(path.read_bytes())))
        return frames


def save_cuts(path: Path, cuts: list[float], duration_sec: float) -> None:
    """Writes a video's cut times and length, in seconds."""
    atomic_write_text(path, json.dumps({"duration_sec": duration_sec, "cuts": cuts}, indent=1))


def load_cuts(path: Path) -> tuple[list[float], float]:
    """Reads the cut times and length saved by `save_cuts`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(cut) for cut in data["cuts"]], float(data["duration_sec"])
