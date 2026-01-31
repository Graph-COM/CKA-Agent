import json
import os
import time
import logging
import threading
import re
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any, List
import torch
from transformers import GenerationConfig

from methods.abstract_method import AbstractJailbreakMethod
from model.model_loader import WhiteBoxModel, BlackBoxModel
from .prompts import *

# Import vLLM for accelerated inference
try:
    from vllm import LLM as VLLMEngine, SamplingParams as VLLMSamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class ActorAttackLLM:
    """Actor Attack's internal AttackLLM for thinking control (similar to PAP/PAIR)."""

    def __init__(self, model_name: str, config: Dict, whitebox_model=None):
        self.config = config or {}
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.tokenizer = None
        self.use_vllm = bool(self.config.get("use_vllm", False))
        self.vllm_kwargs = self.config.get("vllm_kwargs", {}) or {}
        self.hf_token = self.config.get("hf_token")
        self.device = self.config.get("device", "cuda")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 4096))
        self.temperature = float(self.config.get("temperature", 0.7))
        self.top_p = float(self.config.get("top_p", 0.9))
        self.do_sample = bool(self.config.get("do_sample", True))
        self.max_model_len = int(self.config.get("max_model_len", 32768))
        self.enable_thinking = bool(self.config.get("enable_thinking", False))
        self.remove_thinking = bool(self.config.get("remove_thinking", False))

        # Retry configuration (infinite retry with exponential backoff)
        self.rate_limit_backoff_base = float(
            self.config.get("rate_limit_backoff_base", 1.0)
        )
        self.rate_limit_backoff_max = float(
            self.config.get("rate_limit_backoff_max", 60.0)
        )
        self.rate_limit_jitter = float(self.config.get("rate_limit_jitter", 0.2))

        self._vllm_engine = None

        # Use pre-loaded WhiteBoxModel
        if whitebox_model is not None:
            self.logger.info("[ActorAttack] Using pre-loaded WhiteBoxModel")
            self._use_whitebox_model(whitebox_model)
        else:
            raise ValueError("ActorAttackLLM requires a pre-loaded WhiteBoxModel")

    def _use_whitebox_model(self, whitebox_model):
        """Use a pre-loaded WhiteBoxModel."""
        self.tokenizer = whitebox_model.tokenizer
        self.use_vllm = whitebox_model.use_vllm

        if self.use_vllm:
            self._vllm_engine = whitebox_model.vllm_model
            self.logger.info("[ActorAttack] Using vLLM engine from WhiteBoxModel")
        else:
            self.model = whitebox_model.model
            self.logger.info("[ActorAttack] Using HF model from WhiteBoxModel")

        # Copy relevant attributes
        self.model_name = whitebox_model.model_name
        self.device = whitebox_model.device
        self.logger.info(
            f"[ActorAttack] Successfully reused WhiteBoxModel: {self.model_name}"
        )

    def generate(
        self, messages: List[Dict[str, str]], json_format: bool = False
    ) -> Any:
        """
        Generate response with optional JSON format parsing and infinite retry with exponential backoff.
        Will retry indefinitely until successful, handling quota and rate limit errors.

        Args:
            messages: List of messages in chat format
            json_format: Whether to parse response as JSON

        Returns:
            Parsed JSON dict if json_format=True, else string
        """
        import random
        import time

        attempt = 0

        while True:  # Infinite retry until success
            try:
                content = self._chat(messages)

                # Remove thinking tags if configured
                if self.remove_thinking:
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()

                # Parse JSON if requested
                if json_format:
                    try:
                        # Try to extract JSON from markdown code blocks
                        if "```json" in content:
                            json_match = re.search(
                                r"```json\s*(\{.*?\})\s*```", content, re.DOTALL
                            )
                            if json_match:
                                content = json_match.group(1)
                        elif "```" in content:
                            json_match = re.search(
                                r"```\s*(\{.*?\})\s*```", content, re.DOTALL
                            )
                            if json_match:
                                content = json_match.group(1)

                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        # JSON parsing failed - treat as error and retry
                        raise ValueError(f"JSON parsing failed: {e}")

                return content

            except Exception as e:
                attempt += 1
                error_msg = str(e).lower()

                # Check if it's a quota/rate limit error
                is_rate_limit = any(
                    keyword in error_msg
                    for keyword in [
                        "quota",
                        "rate limit",
                        "too many requests",
                        "429",
                        "resource exhausted",
                        "throttl",
                    ]
                )

                if is_rate_limit:
                    # Exponential backoff with jitter for rate limits
                    wait_time = min(
                        self.rate_limit_backoff_base * (2 ** (attempt - 1)),
                        self.rate_limit_backoff_max,
                    )
                    jitter = random.uniform(0, self.rate_limit_jitter * wait_time)
                    total_wait = wait_time + jitter

                    self.logger.warning(
                        f"Rate limit/quota error (attempt {attempt}): {e}. "
                        f"Retrying in {total_wait:.2f}s..."
                    )
                    time.sleep(total_wait)
                elif json_format and "json parsing failed" in error_msg:
                    # JSON parsing error - retry with shorter backoff
                    wait_time = min(1.0 * (1.5 ** (attempt - 1)), 10.0)
                    self.logger.warning(
                        f"JSON parse error (attempt {attempt}): {e}. "
                        f"Regenerating in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # For other errors, still retry but with shorter backoff
                    wait_time = min(2.0 * (1.5 ** (attempt - 1)), 30.0)
                    self.logger.warning(
                        f"Generation error (attempt {attempt}): {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)

                # Continue to retry indefinitely

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Internal chat method."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        # Use vLLM if available
        if self._vllm_engine is not None:
            params = VLLMSamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            outs = self._vllm_engine.generate([prompt_text], params)
            return outs[0].outputs[0].text.strip()
        else:
            # Local HF model processing
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(self.max_model_len, 32768),
            )

            with torch.no_grad():
                # Get model's device from first parameter
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                gen_cfg = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                outputs = self.model.generate(**inputs, generation_config=gen_cfg)
                input_len = inputs["input_ids"].shape[1]
                text = self.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                )

                return text.strip()


class TargetModelWrapper:
    """
    Wrapper for target model to provide chat interface compatible with Actor Attack.

    This wrapper adapts WhiteBoxModel/BlackBoxModel to provide a chat method
    that accepts a list of messages (conversation history).
    """

    def __init__(self, target_model, logger=None):
        """
        Initialize wrapper with target model.

        Args:
            target_model: WhiteBoxModel or BlackBoxModel instance
            logger: Optional logger instance
        """
        self.target_model = target_model
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.tokenizer = getattr(target_model, "tokenizer", None)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response from conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated response string
        """
        try:
            # Convert messages to prompt string
            if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
                # Use tokenizer's chat template if available
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    self.logger.debug(f"Chat template failed, using fallback: {e}")
                    prompt = self._messages_to_prompt(messages)
            else:
                # Fallback to manual formatting
                prompt = self._messages_to_prompt(messages)

            # Generate response using target model
            response = self.target_model.generate(prompt)
            return response

        except Exception as e:
            self.logger.error(f"Target model chat failed: {e}")
            raise

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages to prompt string (fallback method).

        Args:
            messages: List of message dicts

        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")

        # Add generation prompt
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class ActorAttackJudge:
    """
    Internal judge for Actor Attack that evaluates response types.
    Uses Gemini API for evaluation with infinite retry and exponential backoff.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the judge with Gemini configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_name = config.get("model_name", "gemini-2.5-flash")
        self.provider = config.get("provider", "gemini")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.use_proxy = config.get("use_proxy", False)
        self.proxy_api_key = config.get("proxy_api_key")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 512))

        # Retry configuration (infinite retry with exponential backoff)
        self.rate_limit_backoff_base = float(config.get("rate_limit_backoff_base", 1.0))
        self.rate_limit_backoff_max = float(config.get("rate_limit_backoff_max", 60.0))
        self.rate_limit_jitter = float(config.get("rate_limit_jitter", 0.2))

        # Initialize the API client
        if self.provider == "gemini":
            if self.use_proxy:
                # Use OpenAI-compatible proxy
                from openai import OpenAI

                self.client = OpenAI(api_key=self.proxy_api_key, base_url=self.base_url)
            else:
                # Use native Gemini API
                from google import genai

                if not self.api_key:
                    raise ValueError("Gemini api_key is required for official mode")
                if self.base_url:
                    self.client = genai.Client(
                        api_key=self.api_key, http_options={"base_url": self.base_url}
                    )
                else:
                    self.client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.logger.info(f"[ActorAttackJudge] Initialized with {self.model_name}")

    def evaluate_response(self, query: str, response: str) -> str:
        """
        Evaluate a response and return the type classification.
        Will retry indefinitely until successful, handling quota and rate limit errors.

        Args:
            query: The query sent to the target
            response: The response from the target model

        Returns:
            str: Response type ("successful", "rejected", or "failed")
        """
        import random
        import time

        attempt = 0

        while True:  # Infinite retry until success
            try:
                # Construct evaluation prompt using the judge prompt template
                from .prompts import ATTACK_STEP_JUDGE_PROMPT

                eval_prompt = ATTACK_STEP_JUDGE_PROMPT.format(
                    query=query, response=response
                )

                if self.provider == "gemini":
                    if self.use_proxy:
                        # Use OpenAI-compatible API
                        response_obj = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": eval_prompt}],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        eval_response = response_obj.choices[0].message.content
                    else:
                        from google.genai.types import GenerateContentConfig

                        response_obj = self.client.models.generate_content(
                            model=self.model_name,
                            contents=eval_prompt,
                            config=GenerateContentConfig(
                                temperature=self.temperature,
                                max_output_tokens=self.max_tokens,
                            ),
                        )
                        eval_response = ""
                        if hasattr(response_obj, "text") and response_obj.text:
                            eval_response = response_obj.text.strip()
                        elif (
                            hasattr(response_obj, "candidates")
                            and response_obj.candidates
                        ):
                            cand = response_obj.candidates[0]
                            if (
                                hasattr(cand, "content")
                                and cand.content
                                and getattr(cand.content, "parts", None)
                            ):
                                parts = getattr(cand.content, "parts", None)
                                for p in parts:
                                    if hasattr(p, "text") and p.text:
                                        eval_response = p.text.strip()
                                        if eval_response:
                                            break

                        if not eval_response:
                            eval_response = str(response_obj)

                # Parse JSON response
                data = self._parse_json_response(eval_response)
                response_type = data.get("type", "failed")

                return response_type

            except Exception as e:
                attempt += 1
                error_msg = str(e).lower()

                # Check if it's a quota/rate limit error
                is_rate_limit = any(
                    keyword in error_msg
                    for keyword in [
                        "quota",
                        "rate limit",
                        "too many requests",
                        "429",
                        "resource exhausted",
                        "throttl",
                    ]
                )

                if is_rate_limit:
                    # Exponential backoff with jitter for rate limits
                    wait_time = min(
                        self.rate_limit_backoff_base * (2 ** (attempt - 1)),
                        self.rate_limit_backoff_max,
                    )
                    jitter = random.uniform(0, self.rate_limit_jitter * wait_time)
                    total_wait = wait_time + jitter

                    self.logger.warning(
                        f"Rate limit/quota error (attempt {attempt}): {e}. "
                        f"Retrying in {total_wait:.2f}s..."
                    )
                    time.sleep(total_wait)
                else:
                    # For other errors, still retry but with shorter backoff
                    wait_time = min(2.0 * (1.5 ** (attempt - 1)), 30.0)
                    self.logger.warning(
                        f"Evaluation error (attempt {attempt}): {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)

                # Continue to retry indefinitely

    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from response text, handling various formats."""
        import json
        import re

        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
        elif "```" in text:
            json_match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If parsing fails, return default
            self.logger.warning(f"Failed to parse JSON: {text[:200]}")
            return {"type": "failed"}


class ActorAttack(AbstractJailbreakMethod):
    """
    Actor Attack: A multi-turn jailbreak baseline method.

    This method uses role-playing with multiple actors to gradually elicit harmful information
    from the target model through a series of seemingly benign queries.

    Output Format:
    ---------------
    The generate_jailbreak method returns a dict with the following structure:
    {
        'original_query': str,              # Original harmful query
        'jailbreak_prompt': List[Dict],     # Conversation history (multi-turn dialog)
        'response': str,                     # Final response from target model
        'success': bool,                     # Whether attack succeeded internally
        'metadata': {
            'method': 'actor_attack',
            'timestamp': float,
            'processing_time': float,
            'category': str,
            'source': str,
            'actors_tried': int,            # Number of actors attempted
            'successful_actor': str          # Name of successful actor (if any)
        },
        'error': str or None                 # Error message if failed
    }

    Note: The 'jailbreak_prompt' field contains the full conversation history as a list
    of message dicts (unlike single-turn methods that return a string prompt). This is
    compatible with the evaluation framework which only requires 'original_query',
    'response', and 'error' fields for evaluation.

    Performance Optimizations:
    --------------------------
    1. **Dataset-level Batch Processing**: Implements generate_jailbreak_batch() to process
       multiple harmful queries in parallel using ThreadPoolExecutor (when attack model is
       not using vLLM to avoid resource contention).

    2. **Attack LLM Batching**: When attack model uses vLLM, batches query generation for
       all actors (typically 3) in a single vLLM batch call, reducing latency by ~3x for
       the pre-attack phase.

    3. **Thread-safe Intermediate Results**: Uses thread-local storage to track sample
       indices across parallel executions for proper intermediate result saving.

    Configuration:
    --------------
    - actor_num: Number of actors to generate (default: 3)
    - max_retry: Maximum retry attempts for rejected queries (default: 3)
    - save_intermediate: Save intermediate results for debugging (default: True)
    - attack_model: Configuration for the attack model (whitebox/blackbox)
    """

    def __init__(
        self, name: str = "actor_attack", config: Dict[str, Any] = None, model=None
    ):
        """
        Initialize Actor Attack method.

        Args:
            name: Method name
            config: Configuration dictionary containing attack_model settings
            model: Target model instance
        """
        super().__init__(name=name, config=config, model=model)

        # Configuration
        self.actor_num = config.get("actor_num", 3)
        self.max_retry = config.get("max_retry", 3)
        self.save_intermediate = config.get("save_intermediate", True)

        # Output directory injected by main; fallback to current directory
        self.output_dir = getattr(self, "output_dir", os.getcwd())

        # Thread-local storage for tracking sample index
        self._thread_local = threading.local()
        self._sample_locks = defaultdict(threading.Lock)

        # Wrap target model with chat interface
        if model is not None:
            self.target_model_wrapper = TargetModelWrapper(model, self.logger)
            self.logger.info("Target model wrapped with chat interface")
        else:
            self.target_model_wrapper = None
            self.logger.warning("No target model provided")

        # Initialize attack model (aligned with PAIR/PAP structure)
        self.attacker_llm = self._initialize_attack_model()

        # Initialize judge model (Gemini API)
        self.judge = self._initialize_judge()

        self.logger.info(f"Initialized Actor Attack with {self.actor_num} actors")
        self.logger.info(
            f"Attack model: {self.attacker_llm.model_name if self.attacker_llm else 'None'}"
        )
        self.logger.info(
            f"Judge model: {self.judge.model_name if self.judge else 'None'}"
        )
        self.logger.info(f"Output directory: {self.output_dir}")

    def _initialize_attack_model(self):
        """
        Initialize attack model based on configuration (aligned with PAIR/PAP).

        Returns:
            ActorAttackLLM instance for generating attack prompts
        """
        attack_config = self.config.get("attack_model", {})
        attack_type = attack_config.get("type", "whitebox")

        try:
            if attack_type == "whitebox":
                # Use whitebox attack model
                whitebox_config = attack_config.get("whitebox", {})
                model_name = whitebox_config.get("name")

                if not model_name or str(model_name).strip() == "":
                    raise ValueError(
                        f"Model name not specified in config for type: {attack_type}"
                    )

                self.logger.info(f"Initializing whitebox attack model: {model_name}")

                # Create WhiteBoxModel (same as PAP/PAIR)
                whitebox_model = WhiteBoxModel(
                    model_name=model_name, config=whitebox_config
                )

                try:
                    whitebox_model.load(whitebox_config.get("hf_token"))
                    self.logger.info(
                        f"WhiteBoxModel loaded successfully: {whitebox_model.model_name}"
                    )

                    # Create ActorAttackLLM with the loaded WhiteBoxModel
                    attack_llm = ActorAttackLLM(
                        model_name=whitebox_model.model_name,
                        config=whitebox_config,
                        whitebox_model=whitebox_model,
                    )

                    self.logger.info(
                        f"ActorAttackLLM initialized with WhiteBoxModel: {attack_llm.model_name}"
                    )
                    return attack_llm

                except Exception as load_error:
                    self.logger.error(f"Failed to load WhiteBoxModel: {load_error}")
                    raise

            elif attack_type == "blackbox":
                # Use blackbox attack model
                blackbox_config = attack_config.get("blackbox", {})
                self.logger.info(
                    f"Initializing blackbox attack model: {blackbox_config.get('name', 'unknown')}"
                )

                # For blackbox, we could use BlackBoxModel, but for now raise NotImplementedError
                raise NotImplementedError(
                    "Blackbox attack model not yet implemented for Actor Attack"
                )

            else:
                raise ValueError(f"Unknown attack model type: {attack_type}")

        except Exception as e:
            self.logger.error(f"Failed to initialize attack model: {e}")
            raise

    def _initialize_judge(self):
        """
        Initialize judge model (Gemini API) for response evaluation.

        Returns:
            ActorAttackJudge instance for evaluating responses
        """
        judge_config = self.config.get("judge", {})

        if not judge_config:
            self.logger.warning("No judge configuration provided, using defaults")
            judge_config = {
                "model_name": "gemini-2.5-flash",
                "provider": "gemini",
                "temperature": 0.0,
                "max_tokens": 512,
            }

        try:
            judge = ActorAttackJudge(judge_config)
            self.logger.info(f"Initialized ActorAttackJudge with {judge.model_name}")
            return judge
        except Exception as e:
            self.logger.error(f"Failed to initialize judge: {e}")
            raise

    def generate_jailbreak(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a jailbreak attempt using Actor Attack method.

        Args:
            query: The harmful query to attack
            **kwargs: Additional parameters (category, source)

        Returns:
            Dict containing attack results and metadata
        """
        start_time = time.time()

        result = {
            "original_query": query,
            "jailbreak_prompt": None,
            "response": None,
            "success": False,
            "metadata": {
                "method": self.name,
                "timestamp": start_time,
                "processing_time": 0,
                "category": kwargs.get("category", "unknown"),
                "source": kwargs.get("source", "unknown"),
                "actors_tried": 0,
                "successful_actor": None,
            },
            "error": None,
        }

        try:
            # Check if target model wrapper exists
            if self.target_model_wrapper is None:
                raise ValueError(
                    "Target model not initialized. Actor Attack requires a target model."
                )

            # Pre-attack phase: Extract harm target and generate actors
            pre_attack_data = self._infer_single(query, actor_num=self.actor_num)

            if self.save_intermediate:
                self._save_intermediate(
                    "pre_attack",
                    {
                        "original_query": query,
                        "harm_target": pre_attack_data["harm_target"],
                        "query_details": pre_attack_data["query_details"],
                        "num_actors": len(pre_attack_data["actors"]),
                        "actors": [
                            {
                                "actor_name": a["actor"]["actor_name"],
                                "relationship": a["actor"]["relationship"],
                            }
                            for a in pre_attack_data["actors"]
                        ],
                    },
                )

            final_response = ""
            final_dialog = []
            successful_actor = None

            # Track last dialog and response for fallback
            last_dialog = []
            last_response = ""

            # In-attack phase: Try each actor
            for actor_idx, actor_data in enumerate(pre_attack_data["actors"]):
                result["metadata"]["actors_tried"] = actor_idx + 1

                dialog_hist, skip_actor = self._call_multi(
                    self.target_model_wrapper,
                    actor_data["queries"],
                    pre_attack_data["query_details"],
                )

                if self.save_intermediate:
                    self._save_intermediate(
                        "actor_attempt",
                        {
                            "actor_index": actor_idx,
                            "actor_name": actor_data["actor"]["actor_name"],
                            "num_queries": len(actor_data["queries"]),
                            "dialog_length": len(dialog_hist),
                            "skipped": skip_actor,
                        },
                    )

                if skip_actor:
                    continue

                # Summary phase: Request final harmful output
                response, dialog = self._summary(
                    pre_attack_data["query_details"],
                    dialog_hist,
                    self.target_model_wrapper,
                )

                # Track for fallback
                last_dialog = dialog
                last_response = response

                # Judge if attack succeeded
                if dialog and len(dialog) >= 2:
                    response_type = self._step_judge(dialog[-2]["content"], response)

                    if self.save_intermediate:
                        self._save_intermediate(
                            "summary_attempt",
                            {
                                "actor_index": actor_idx,
                                "response_type": response_type,
                                "response_length": len(response),
                                "dialog_turns": len(dialog) // 2,
                            },
                        )

                    if response_type == "successful":
                        final_dialog = dialog
                        final_response = response
                        successful_actor = actor_data["actor"]["actor_name"]
                        result["metadata"]["successful_actor"] = successful_actor
                        break

            # Store results
            if final_dialog:
                result["jailbreak_prompt"] = final_dialog
                result["response"] = final_response
                result["success"] = bool(successful_actor)
            else:
                # Fallback if no actor succeeded - use last attempt
                result["jailbreak_prompt"] = last_dialog
                result["response"] = last_response
                result["success"] = False

            # Update stats
            self.update_stats(success=result["success"], error=False)

        except Exception as e:
            error_msg = f"Error in Actor Attack: {str(e)}"
            self.logger.error(error_msg)
            import traceback

            self.logger.error(traceback.format_exc())

            result["error"] = error_msg
            result["success"] = False
            self.update_stats(success=False, error=True)

        # Calculate processing time
        result["metadata"]["processing_time"] = time.time() - start_time

        return result

    def prepare_prompt(self, query: str, **kwargs) -> str:
        """
        Prepare prompt for Actor Attack (not used in multi-turn attack).

        Args:
            query: Original harmful query

        Returns:
            The original query (Actor Attack doesn't use static prompt transformation)
        """
        return query

    def generate_jailbreak_batch(
        self, queries: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Dataset-level batch processing for multiple harmful queries.

        When attack LLM uses vLLM, process sequentially to avoid resource contention.
        Otherwise, can use threading for parallel processing.

        Args:
            queries: List of harmful queries to attack
            **kwargs: Additional parameters (category, source, target_str, dataset_key, base_index)

        Returns:
            List of attack results
        """
        batch_size = getattr(self.model, "batch_size", 1)

        # Check if attack LLM uses vLLM (disable parallel processing if so)
        try:
            if hasattr(self, "attacker_llm") and getattr(
                self.attacker_llm, "use_vllm", False
            ):
                if batch_size != 1:
                    self.logger.info(
                        f"[ActorAttack] attack_llm.use_vllm=True -> forcing batch_size from {batch_size} to 1 (disable threading)"
                    )
                batch_size = 1
        except Exception:
            pass

        # Get global base index and dataset key for intermediate results
        base_index = int(kwargs.get("base_index", 0))
        dataset_key = kwargs.get("dataset_key", "unknown")

        # Extract additional metadata if provided
        categories = kwargs.get("category", ["unknown"] * len(queries))
        sources = kwargs.get("source", ["unknown"] * len(queries))
        target_strs = kwargs.get("target_str", [""] * len(queries))

        # Ensure lists have same length as queries
        if not isinstance(categories, list):
            categories = [categories] * len(queries)
        if not isinstance(sources, list):
            sources = [sources] * len(queries)
        if not isinstance(target_strs, list):
            target_strs = [target_strs] * len(queries)

        results: List[Dict[str, Any]] = []

        if batch_size <= 1:
            # Sequential processing (default for vLLM attack models)
            self.logger.info(
                f"[ActorAttack] Processing {len(queries)} queries sequentially"
            )
            for i, q in enumerate(queries):
                # Set thread-local storage for intermediate results tracking
                try:
                    self._thread_local.sample_index = base_index + i
                    self._thread_local.dataset_key = dataset_key
                except Exception:
                    pass

                result = self.generate_jailbreak(
                    q,
                    category=categories[i],
                    source=sources[i],
                    target_str=target_strs[i],
                )
                results.append(result)
            return results

        # Batch processing with threading (when attack LLM doesn't use vLLM)
        self.logger.info(
            f"[ActorAttack] Processing {len(queries)} queries in batches of {batch_size}"
        )

        import concurrent.futures

        def process_single_query(idx, query, category, source, target_str):
            """Process a single query with thread-local storage."""
            try:
                # Set thread-local storage for this thread
                self._thread_local.sample_index = base_index + idx
                self._thread_local.dataset_key = dataset_key

                result = self.generate_jailbreak(
                    query, category=category, source=source, target_str=target_str
                )
                return (idx, result)
            except Exception as e:
                self.logger.error(f"[ActorAttack] Error processing query {idx}: {e}")
                return (
                    idx,
                    {
                        "original_query": query,
                        "error": str(e),
                        "success": False,
                        "metadata": {
                            "method": self.name,
                            "category": category,
                            "source": source,
                            "error": str(e),
                        },
                    },
                )

        # Process in batches with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for i, (q, cat, src, tgt) in enumerate(
                zip(queries, categories, sources, target_strs)
            ):
                future = executor.submit(process_single_query, i, q, cat, src, tgt)
                futures.append(future)

            # Collect results in order
            indexed_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    indexed_results.append(future.result())
                except Exception as e:
                    self.logger.error(f"[ActorAttack] Future execution error: {e}")

            # Sort by index to maintain order
            indexed_results.sort(key=lambda x: x[0])
            results = [result for _, result in indexed_results]

        return results

    def _infer_single(self, org_query, actor_num):
        harm_target, query_details = self._extract_harm_target(org_query)
        actors, _ = self._get_actors(harm_target, actor_num)
        data_list = []

        # Optimize: Batch query generation for all actors if using vLLM
        if self.attacker_llm and getattr(self.attacker_llm, "use_vllm", False):
            # Batch process query generation for all actors
            data_list = self._get_init_queries_batch(harm_target, actors)
        else:
            # Sequential processing (original behavior)
            for actor in actors:
                queries, _ = self._get_init_queries(harm_target, actor)
                data_list.append({"actor": actor, "queries": queries})

        return {
            "instruction": org_query,
            "harm_target": harm_target,
            "query_details": query_details,
            "actors": data_list,
        }

    def _extract_harm_target(self, org_query):
        prompt = EXTRACT_PROMPT.format(org_query=org_query)
        messages = [{"role": "user", "content": prompt}]
        data = self.attacker_llm.generate(messages, json_format=True)
        return data["target"], data["details"]

    def _get_actors(self, harm_target, actor_num):
        network_prompt = NETWORK_PROMPT.format(harm_target=harm_target)
        messages = [{"role": "user", "content": network_prompt}]
        response = self.attacker_llm.generate(messages, json_format=False)
        messages.append({"role": "assistant", "content": response})
        num_string = "10 actors" if actor_num > 10 else f"{actor_num} actors"
        actor_prompt = ACTOR_PROMPT.format(num_string=num_string)
        actors = []
        messages.append({"role": "user", "content": actor_prompt})
        data = self.attacker_llm.generate(messages, json_format=True)
        messages.append({"role": "assistant", "content": json.dumps(data)})
        for item in data["actors"]:
            if item["actor_name"] not in [
                actor_item["actor_name"] for actor_item in actors
            ]:
                actors.append(item)
        messages = messages[:-2]
        if len(actors) >= actor_num:
            return actors[:actor_num], messages
        messages.append({"role": "user", "content": MORE_ACTOR_PROMPT})
        response = self.attacker_llm.generate(messages, json_format=False)
        return actors, messages

    def _get_init_queries(self, harm_target, actor):
        actor_name = actor["actor_name"]
        relationship = actor["relationship"]
        query_prompt = QUERIES_PROMPT.format(
            harm_target=harm_target, actor_name=actor_name, relationship=relationship
        )
        messages = [{"role": "user", "content": query_prompt}]
        query_resp = self.attacker_llm.generate(messages, json_format=False)
        format_prompt = JSON_FORMAT_QUESTION_PROMPT.format(resp=query_resp)
        messages = [{"role": "user", "content": format_prompt}]
        data = self.attacker_llm.generate(messages, json_format=True)
        queries = [item["question"] for item in data["questions"]]
        return queries, query_resp

    def _get_init_queries_batch(self, harm_target, actors):
        """
        Batch version of _get_init_queries for vLLM optimization.

        Generates initial queries for all actors in a batch to leverage vLLM's
        parallel processing capabilities.

        Args:
            harm_target: The extracted harm target
            actors: List of actor dictionaries

        Returns:
            List of dicts with actor and queries
        """
        if not actors:
            return []

        # Step 1: Batch generate query responses for all actors
        query_prompts = []
        for actor in actors:
            actor_name = actor["actor_name"]
            relationship = actor["relationship"]
            query_prompt = QUERIES_PROMPT.format(
                harm_target=harm_target,
                actor_name=actor_name,
                relationship=relationship,
            )
            query_prompts.append(query_prompt)

        # Use vLLM's batch processing via _chat_batch
        query_responses = self._chat_batch_attacker(query_prompts)

        # Step 2: Batch format responses to JSON
        format_prompts = []
        for query_resp in query_responses:
            format_prompt = JSON_FORMAT_QUESTION_PROMPT.format(resp=query_resp)
            format_prompts.append(format_prompt)

        # Batch generate JSON formatted data
        json_responses = self._chat_batch_attacker(format_prompts, json_format=True)

        # Step 3: Compile results
        data_list = []
        for actor, json_data in zip(actors, json_responses):
            if json_data and "questions" in json_data:
                queries = [item["question"] for item in json_data["questions"]]
            else:
                # Fallback to empty list if JSON parsing failed
                self.logger.warning(
                    f"Failed to parse queries for actor {actor.get('actor_name', 'unknown')}"
                )
                queries = []
            data_list.append({"actor": actor, "queries": queries})

        return data_list

    def _chat_batch_attacker(
        self, prompts: List[str], json_format: bool = False
    ) -> List[Any]:
        """
        Helper method to batch chat with attacker LLM.

        Args:
            prompts: List of prompt strings
            json_format: Whether to parse responses as JSON

        Returns:
            List of responses (strings or parsed JSON dicts)
        """
        if not prompts:
            return []

        # Convert prompts to message format
        message_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # Check if attacker LLM has vLLM engine for true batch processing
        if (
            hasattr(self.attacker_llm, "_vllm_engine")
            and self.attacker_llm._vllm_engine is not None
        ):
            # Use vLLM batch processing
            prompt_texts = []
            for messages in message_list:
                prompt_text = self.attacker_llm.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=getattr(
                        self.attacker_llm, "enable_thinking", False
                    ),
                )
                prompt_texts.append(prompt_text)

            # Batch generate with vLLM
            params = VLLMSamplingParams(
                max_tokens=self.attacker_llm.max_new_tokens,
                temperature=self.attacker_llm.temperature,
                top_p=self.attacker_llm.top_p,
            )
            outputs = self.attacker_llm._vllm_engine.generate(prompt_texts, params)

            responses = []
            for output in outputs:
                content = output.outputs[0].text.strip()

                # Remove thinking tags if configured
                if getattr(self.attacker_llm, "remove_thinking", False):
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()

                # Parse JSON if requested
                if json_format:
                    try:
                        # Try to extract JSON from markdown code blocks
                        if "```json" in content:
                            json_match = re.search(
                                r"```json\s*(\{.*?\})\s*```", content, re.DOTALL
                            )
                            if json_match:
                                content = json_match.group(1)
                        elif "```" in content:
                            json_match = re.search(
                                r"```\s*(\{.*?\})\s*```", content, re.DOTALL
                            )
                            if json_match:
                                content = json_match.group(1)

                        responses.append(json.loads(content))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON in batch: {e}")
                        responses.append({})
                else:
                    responses.append(content)

            return responses
        else:
            # Fallback to sequential processing if not using vLLM
            responses = []
            for messages in message_list:
                response = self.attacker_llm.generate(messages, json_format=json_format)
                responses.append(response)
            return responses

    def _call_multi(self, target_llm, queries, query_details):
        """
        Execute multi-turn conversation with target model.

        Args:
            target_llm: Target model instance
            queries: List of queries to ask
            query_details: Details about the query delivery type

        Returns:
            Tuple of (dialog_history, skip_actor_flag)
        """
        dialog_hist = []
        N_retry = self.max_retry

        original_queries = list(queries)

        for i in range(len(original_queries)):
            query_to_try = original_queries[i]
            retries_failed = True
            for attempt in range(N_retry + 1):
                current_hist = dialog_hist + [{"role": "user", "content": query_to_try}]

                # Generate response from target model wrapper (guaranteed to have chat method)
                try:
                    response = target_llm.chat(current_hist)
                except Exception as e:
                    self.logger.error(f"Error calling target model: {e}")
                    return dialog_hist, True

                response_type = self._step_judge(query_to_try, response)
                if response_type == "successful":
                    dialog_hist.append({"role": "user", "content": query_to_try})
                    dialog_hist.append({"role": "assistant", "content": response})
                    retries_failed = False
                    break
                elif response_type == "unknown":
                    return dialog_hist, True
                else:  # response_type == "rejective"
                    if attempt < N_retry:
                        query_to_try = self._rewrite_query(
                            original_queries[i],
                            response,
                            original_queries,
                            query_details,
                        )
                    else:
                        return dialog_hist, True
            if retries_failed:
                return dialog_hist, True

        return dialog_hist, False

    def _step_judge(self, query, response):
        """Evaluate response using the Gemini judge API."""
        return self.judge.evaluate_response(query, response)

    def _rewrite_query(self, query, last_response, queries, query_details):
        queries_str = "".join(f"{j + 1}. {q}\n" for j, q in enumerate(queries))
        prompt = ATTACK_MODIFY_PROMPT.format(
            queries=queries_str,
            query=query,
            output_type=query_details["delivery_type"],
            previous_response=last_response,
        )
        messages = [{"role": "user", "content": prompt}]
        new_query = self.attacker_llm.generate(messages, json_format=False)
        return new_query

    def _summary(self, query_details, dialog_hist, target_llm):
        """
        Request summary/final harmful output from target model.

        Args:
            query_details: Details about the query
            dialog_hist: Conversation history
            target_llm: Target model instance

        Returns:
            Tuple of (response, final_dialog)
        """
        detail_query = SUMMARY_WITH_DETAILS_PROMPT.format(
            delivery_type=query_details["delivery_type"],
            other_details=query_details["other_details"],
        )
        type_query = SUMMARY_WITHOUT_DETAILS_PROMPT.format(
            delivery_type=query_details["delivery_type"]
        )

        summary_queries = [detail_query, type_query]
        if not query_details.get("other_details"):
            summary_queries = [type_query]

        for summary_query in summary_queries:
            current_hist = dialog_hist + [{"role": "user", "content": summary_query}]

            # Generate response from target model wrapper (guaranteed to have chat method)
            try:
                resp = target_llm.chat(current_hist)
            except Exception as e:
                self.logger.error(f"Error calling target model in summary: {e}")
                # Return last response or empty if none
                resp = resp if "resp" in locals() else ""
                final_dialog = current_hist + [{"role": "assistant", "content": resp}]
                return resp, final_dialog

            response_type = self._step_judge(summary_query, resp)
            if response_type == "successful":
                final_dialog = current_hist + [{"role": "assistant", "content": resp}]
                return resp, final_dialog

        final_dialog = current_hist + [{"role": "assistant", "content": resp}]
        return resp, final_dialog

    def _save_intermediate(self, stage: str, data: Dict[str, Any]) -> None:
        """
        Save intermediate results for debugging/analysis.

        Args:
            stage: Attack stage name (pre_attack/actor_attempt/summary_attempt)
            data: Stage-specific data to save
        """
        try:
            sample_index = getattr(self._thread_local, "sample_index", 0)
            dataset_key = getattr(self._thread_local, "dataset_key", "unknown")

            with self._sample_locks[sample_index]:
                os.makedirs(self.output_dir, exist_ok=True)
                inter_path = os.path.join(
                    self.output_dir, f"inter_result_sample_{sample_index}.json"
                )

                payload = {
                    "stage": stage,
                    "sample_index": sample_index,
                    "dataset_key": dataset_key,
                    "timestamp": time.time(),
                    "data": data,
                }

                with open(inter_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")
