from pathlib import Path
from typing import Iterable

from doc_page_extractor.types import (
    DeepSeekOCRModel,
    DeepSeekOCRSize,
    ExtractionContext,
)
from readerwriterlock import rwlock


class DeepSeekOCRMlxVlmModel(DeepSeekOCRModel):
    """
    使用 mlx-vlm 库直接加载和推理 mlx-community/DeepSeek-OCR-8bit 模型。

    model_path: 本地模型缓存路径，如果为 None 则从 Hugging Face 下载。
    local_only: 如果为 True，仅使用本地模型，不从网络下载。
    """

    def __init__(
        self,
        # model_name: str = "mlx-community/DeepSeek-OCR-8bit",
        model_name: str = "mlx-community/DeepSeek-OCR-2-bf16",
        model_path: Path | None = None,
        local_only: bool = False,
        enable_devices_numbers: Iterable[int] | None = None,
    ) -> None:
        if local_only and model_path is None:
            raise ValueError("model_path must be provided when local_only is True")

        self._model_name = model_name
        self._model_path = model_path
        self._local_only = local_only
        self._model = None
        self._processor = None
        self._config = None
        self._rwlock = rwlock.RWLockFair()

    def download(self, revision: str | None) -> None:
        """Download model from Hugging Face Hub to cache directory."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub package is required") from exc

        cache_dir = str(self._model_path) if self._model_path else None
        with self._rwlock.gen_wlock():
            snapshot_download(
                repo_id=self._model_name,
                repo_type="model",
                revision=revision,
                force_download=True,
                cache_dir=cache_dir,
            )

    def load(self) -> None:
        """Load model and processor into memory."""
        self._ensure_model_and_processor()

    def unload(self) -> None:
        """Unload model and processor from memory."""
        with self._rwlock.gen_wlock():
            self._model = None
            self._processor = None
            self._config = None

    def generate(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,  # noqa: ARG002
        size: DeepSeekOCRSize,  # noqa: ARG002
        context: ExtractionContext | None,  # noqa: ARG002
        device_number: int | None,  # noqa: ARG002
    ) -> str:
        """
        Generate OCR result using mlx-vlm model.

        Args:
            prompt: The OCR prompt/instruction.
            image_path: Path to the image file.
            output_path: Path to save output files.
            size: Model size variant (not used in mlx-vlm version).
            context: Extraction context for interruption handling.
            device_number: GPU device number (not applicable for mlx).

        Returns:
            OCR result text.
        """
        model, processor, config = self._ensure_model_and_processor()

        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise RuntimeError(
                "mlx-vlm package is required for DeepSeekOCRMlxVlmModel"
            ) from exc

        # Prepare image path as list
        image = [str(image_path)]

        with self._rwlock.gen_rlock():
            try:
                # Apply chat template with num_images=1 for single image OCR
                formatted_prompt = apply_chat_template(
                    processor, config, prompt, num_images=1
                )

                # Normalize image token to string
                image_token = getattr(processor, "image_token", "<image>")
                token_str = str(image_token)

                # formatted_prompt may be a string, a list of messages, or other structure.
                # Handle string case first (most common), otherwise try to sanitize list/dict messages
                if isinstance(formatted_prompt, str):
                    image_count = formatted_prompt.count(token_str)
                    if image_count > 1:
                        parts = formatted_prompt.split(token_str)
                        formatted_prompt = token_str.join([parts[0], parts[-1]])
                elif isinstance(formatted_prompt, list):
                    # Iterate through list items (could be dict messages or strings) and
                    # ensure only the first occurrence of the image token remains.
                    first_seen = False
                    for i, item in enumerate(formatted_prompt):
                        if (
                            isinstance(item, dict)
                            and "content" in item
                            and isinstance(item["content"], str)
                        ):
                            content = item["content"]
                            if token_str in content:
                                if not first_seen:
                                    parts = content.split(token_str)
                                    if len(parts) > 1:
                                        content = token_str.join([parts[0], parts[-1]])
                                    first_seen = True
                                else:
                                    content = content.replace(token_str, "")
                                formatted_prompt[i]["content"] = content
                        elif isinstance(item, str):
                            content = item
                            if token_str in content:
                                if not first_seen:
                                    parts = content.split(token_str)
                                    if len(parts) > 1:
                                        content = token_str.join([parts[0], parts[-1]])
                                    first_seen = True
                                else:
                                    content = content.replace(token_str, "")
                                formatted_prompt[i] = content  # type: ignore
                    # fall through: if not string/list, leave as-is

                # Generate response using mlx_vlm.generate
                result = generate(
                    model,
                    processor,
                    formatted_prompt,  # type: ignore
                    image,
                    max_tokens=8192,
                    verbose=False,
                )

                # Extract text from GenerationResult object
                # mlx_vlm.generate returns a GenerationResult, extract the text
                output_text = str(result)
                if hasattr(result, "text"):
                    output_text = result.text
                elif hasattr(result, "__str__"):
                    output_text = str(result)

                return output_text

            except Exception as exc:
                raise RuntimeError(f"Failed to generate OCR result: {exc}") from exc

    def _ensure_model_and_processor(self) -> tuple:
        """Ensure model, processor, and config are loaded, implementing double-check locking."""
        with self._rwlock.gen_rlock():
            if (
                self._model is not None
                and self._processor is not None
                and self._config is not None
            ):
                return self._model, self._processor, self._config

        with self._rwlock.gen_wlock():
            # Double-check after acquiring write lock
            if (
                self._model is not None
                and self._processor is not None
                and self._config is not None
            ):
                return self._model, self._processor, self._config

            try:
                from mlx_vlm import load
                from mlx_vlm.utils import load_config
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-vlm package is required for DeepSeekOCRMlxVlmModel"
                ) from exc

            # Determine model path
            model_path_or_name = self._model_name
            if self._local_only:
                model_path_or_name = self._find_local_model_path()
                if model_path_or_name is None:
                    raise ValueError(
                        f"Local model not found at {self._model_path}. "
                        f"Please run download() first to download the model."
                    )

            # Load model, processor, and config
            try:
                self._model, self._processor = load(model_path_or_name)
                self._config = load_config(model_path_or_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load model from {model_path_or_name}: {exc}"
                ) from exc

            return self._model, self._processor, self._config

    def _find_local_model_path(self) -> str | None:
        """Find local model path in Hugging Face cache structure."""
        if self._model_path is None:
            return None

        # Hugging Face cache structure: cache_dir/models--{org}--{model}/snapshots/{hash}/
        cache_model_dir = (
            self._model_path / "models--mlx-community--DeepSeek-OCR-2-bf16"
        )
        if not cache_model_dir.exists():
            return None

        # Try to find the latest snapshot
        snapshots_dir = cache_model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            return None

        latest_snapshot = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
        return str(latest_snapshot)
