"""LLM API client wrapper with retry and token tracking."""

import time
import random
import logging
from dataclasses import dataclass, field
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class TokenReservation:
    timestamp: float
    tokens: int


class RollingTokenRateLimiter:
    """Shared rolling-window token-per-minute limiter for LLM calls."""

    def __init__(
        self,
        tokens_per_minute: int | None,
        window_seconds: int = 60,
        time_fn=time.time,
        sleep_fn=time.sleep,
    ):
        self.tokens_per_minute = int(tokens_per_minute or 0)
        self.window_seconds = window_seconds
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self._reservations: list[TokenReservation] = []

    @property
    def enabled(self) -> bool:
        return self.tokens_per_minute > 0

    def acquire(self, estimated_tokens: int) -> TokenReservation | None:
        """Wait until the rolling window has room, then reserve estimated tokens."""
        if not self.enabled:
            return None

        tokens = max(int(estimated_tokens), 1)
        while True:
            now = self.time_fn()
            self._prune(now)
            used = sum(item.tokens for item in self._reservations)
            if used == 0 or used + tokens <= self.tokens_per_minute:
                reservation = TokenReservation(timestamp=now, tokens=tokens)
                self._reservations.append(reservation)
                return reservation

            oldest = min(item.timestamp for item in self._reservations)
            wait = max(oldest + self.window_seconds - now, 0.001)
            logger.info(
                "LLM TPM limiter sleeping %.1fs "
                "(used=%s, requested=%s, budget=%s)",
                wait,
                used,
                tokens,
                self.tokens_per_minute,
            )
            self.sleep_fn(wait)

    def record_actual(self, reservation: TokenReservation | None, actual_tokens: int) -> None:
        """Replace reserved tokens with actual API usage when the provider returns it."""
        if reservation is None or actual_tokens <= 0:
            return
        reservation.tokens = max(int(actual_tokens), 1)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._reservations = [
            item for item in self._reservations
            if item.timestamp > cutoff
        ]


@dataclass
class TokenTracker:
    total_input: int = 0
    total_output: int = 0
    total_calls: int = 0
    call_log: list = field(default_factory=list)

    def record(self, resp: LLMResponse, label: str = "") -> None:
        self.total_input += resp.input_tokens
        self.total_output += resp.output_tokens
        self.total_calls += 1
        if label:
            self.call_log.append({
                "label": label,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            })

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "total_calls": self.total_calls,
        }

    def reset(self) -> None:
        self.total_input = 0
        self.total_output = 0
        self.total_calls = 0
        self.call_log.clear()


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3,
        rate_limiter: RollingTokenRateLimiter | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter
        self.tracker = TokenTracker()
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)

    def chat(
        self,
        messages: list[dict],
        stop: list[str] | None = None,
        label: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send chat completion request with retry."""
        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            reservation = None
            try:
                estimated_tokens = estimate_chat_tokens(messages, mt)
                if self.rate_limiter is not None:
                    reservation = self.rate_limiter.acquire(estimated_tokens)

                t0 = time.time()
                # Disable thinking mode for reasoning models (e.g., Qwen3.5)
                extra = {}
                if "qwen3" in self.model.lower() or "qwq" in self.model.lower():
                    extra["extra_body"] = {"enable_thinking": False}

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=mt,
                    stop=stop,
                    **extra,
                )
                latency = (time.time() - t0) * 1000

                content = resp.choices[0].message.content or ""
                usage = resp.usage
                result = LLMResponse(
                    content=content.strip(),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency,
                )
                if self.rate_limiter is not None:
                    self.rate_limiter.record_actual(
                        reservation,
                        result.input_tokens + result.output_tokens,
                    )
                self.tracker.record(result, label=label)
                return result

            except Exception as e:
                wait = _retry_wait_seconds(e, attempt)
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {wait}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    logger.error(f"LLM call failed after {self.max_retries} attempts")
                    raise

    def complete_text(
        self,
        prompt: str,
        stop: list[str] | None = None,
        label: str = "",
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Convenience: single prompt string → response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, stop=stop, label=label, max_tokens=max_tokens).content


def estimate_chat_tokens(messages: list[dict], max_tokens: int) -> int:
    """Estimate total tokens for pre-request throttling without provider tokenizer access."""
    content_chars = 0
    for message in messages:
        content_chars += len(str(message.get("role", "")))
        content_chars += len(str(message.get("content", "")))
    prompt_tokens = max(1, (content_chars + 3) // 4 + 4 * len(messages))
    return prompt_tokens + max(int(max_tokens or 0), 0)


def _retry_wait_seconds(error: Exception, attempt: int) -> int:
    message = str(error).lower()
    if "429" in message or "rate limit" in message or "tpm limit" in message:
        return 30 * (2 ** attempt)
    return 2 ** (attempt + 1)
