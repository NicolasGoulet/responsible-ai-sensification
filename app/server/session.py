"""session.py: Pipeline parameter dataclass and session state."""
import asyncio
from dataclasses import dataclass, field


@dataclass
class PipelineParams:
    prompt: str = "Hello world"
    model: str = "google/gemma-3-1b-pt"
    layer: int = 22
    width: str = "65k"
    l0: str = "medium"
    max_tokens: int = 200
    strategy: str = "identity"   # "identity" | "cluster"
    clusters: int = 8
    loop: bool = False
    mode: str = "timed"          # "timed" | "sustain"

    def update(self, **kwargs) -> None:
        """Merge a dict of partial params into this instance."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class PipelineSession:
    params: PipelineParams = field(default_factory=PipelineParams)
    task: asyncio.Task | None = None

    def is_running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
