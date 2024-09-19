from typing import Any, AsyncGenerator, List, Dict, Union, Optional, get_type_hints
from tenacity import retry, stop_after_attempt, wait_random_exponential
from instructor.function_calls import OpenAISchema, Mode
from openai.types.chat.chat_completion import Choice
from litellm import (
    acompletion as aget_litellm_chat_completion,
    embedding as get_litellm_embedding,
    CustomStreamWrapper,
    get_llm_provider,
    EmbeddingResponse,
    ModelResponse,
)
from openai.types.chat import ChatCompletion
from fastapi import HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from functools import wraps
import instructor
import traceback
import asyncio
import json
import ast
import os

from utils.llm_schema import (
    LiteLLMKwargs,
    MultiGenerationsResponse,
    Provider,
    ChatMessage,
    MessageRole,
    ParseType,
)
from prompt import BasePromptFunction
from base_error import ServerError
from logging import logger


class LLMClient:
    def __init__(self):
        self.chat_model = None
        self.embed_model = None
        self._patch_structure_completion()

        key = LLMClient._openai_key()
        self._openai_client = AsyncOpenAI(api_key=key)
        self.client = instructor.from_openai(client=self._openai_client)


    ################################################################################
    #                                                                              #
    #                                 LLM Completion                               #
    #                                                                              #
    ################################################################################

    def embed(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        litellm_kwargs: LiteLLMKwargs = LiteLLMKwargs(),
        **additional_kwargs: Any,
    ) -> List[List[float]]:
        # Merge litellm_kwargs with additional_kwargs
        all_kwargs = LiteLLMKwargs(
            **(litellm_kwargs.dict(exclude_none=True) | additional_kwargs)
        )

        response = self._get_embeddings_response(
            texts=texts,
            model=model,
            litellm_kwargs=all_kwargs,
        )

        # Return the embeddings as a list of lists of floats
        try:
            if litellm_kwargs.aws_access_key_id is not None:
                # bedrock can't handle batches
                return [result["embedding"] for result in response.data]
            return [result.embedding for result in response.data]
        except AttributeError:
            return [result["embedding"] for result in response.data]

    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        litellm_kwargs: LiteLLMKwargs = LiteLLMKwargs(),
        **additional_kwargs: Any,
    ) -> ModelResponse:
        # Merge litellm_kwargs with additional_kwargs
        all_kwargs = LiteLLMKwargs(
            **(litellm_kwargs.dict(exclude_none=True) | additional_kwargs)
        )

        try:
            # Make sure stream is false for non-streaming completion
            all_kwargs.stream = False

            response = await self._aget_chat_completion_response(
                messages=messages,
                model=model,
                litellm_kwargs=all_kwargs,
            )

            return response
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise ServerError.LLM_API_ERROR.as_exception(
                f"LLM API Completion Error, details: {str(e)}",
            )

    async def astream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        litellm_kwargs: LiteLLMKwargs = LiteLLMKwargs(),
        **additional_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        # Merge litellm_kwargs with additional_kwargs
        all_kwargs = LiteLLMKwargs(
            **(litellm_kwargs.dict(exclude_none=True) | additional_kwargs)
        )

        try:
            # Make sure stream is true for streaming completion
            all_kwargs.stream = True

            stream = await self._aget_chat_completion_response(
                messages=messages,
                model=model,
                litellm_kwargs=all_kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise ServerError.LLM_API_ERROR.as_exception(
                f"LLM API Streaming Error, details: {str(e)}",
            )

    async def astructure_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        response_model: Optional[BaseModel] = None,
        litellm_kwargs: LiteLLMKwargs = LiteLLMKwargs(),
        **additional_kwargs: Any,
    ) -> ModelResponse:
        """
        Steer the LLM to output answer in specific format conformed to response_model.
        response_model is a pydantic model that describes the expected output format.
        For example:
            class Response(BaseModel):
                answer: str
        More details: https://python.useinstructor.com/
        """
        # Merge litellm_kwargs with additional_kwargs
        all_kwargs = LiteLLMKwargs(
            **(litellm_kwargs.dict(exclude_none=True) | additional_kwargs)
        )

        try:
            response = await self._astructure_completion_func(
                messages=messages,
                model=model,
                response_model=response_model,
                **all_kwargs.model_dump(exclude_none=True),
            )

            return response
        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"Error: {str(e)}")
            raise ServerError.LLM_API_ERROR.as_exception(
                f"LLM API Steering Error, details: {str(e)}",
            )

    async def astructure_stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        response_model: Optional[BaseModel] = None,
        litellm_kwargs: LiteLLMKwargs = LiteLLMKwargs(),
        **additional_kwargs: Any,
    ) -> AsyncGenerator:
        """
        Stream the LLM output to response_model.
        More details: https://python.useinstructor.com/hub/partial_streaming/
        """
        # Merge litellm_kwargs with additional_kwargs
        all_kwargs = LiteLLMKwargs(
            **(litellm_kwargs.dict(exclude_none=True) | additional_kwargs)
        )
        response = None

        try:
            # Make sure stream is true for partial streaming
            all_kwargs.stream = True

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                **all_kwargs.model_dump(exclude_none=True),
            )

            async for chunk in response:
                if chunk:
                    yield chunk

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise ServerError.LLM_API_ERROR.as_exception(
                f"LLM API Partial Steering Error, details: {str(e)}",
            )
        finally:
            if response:
                await response.aclose()

    ################################################################################
    #                                                                              #
    #                               Prompt Wrappers                                #
    #                                                                              #
    ################################################################################

    async def acompletion_with_prompt(
        self,
        prompt_func: BasePromptFunction,
        parse_type: ParseType | None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Wrapper around acompletion to allow for passing in a prompt function.
        """
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt_func.system()),
            ChatMessage(role=MessageRole.USER, content=prompt_func.user()),
        ]

        response = await self.client.chat.completions.create(
            messages=messages,
            **kwargs,
        )
        output = response.choices[0].message.content
        if parse_type == ParseType.JSON:
            output = output.strip("`").strip("json")
            output = json.loads(output)
        elif parse_type == ParseType.LIST:
            output = ast.literal_eval(output)
        elif parse_type == ParseType.RAW:
            return response

        return output

    def astream_completion_with_prompt(
        self,
        prompt_func: BasePromptFunction,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Wrapper around astream_completion to allow for passing in a prompt function.
        """
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt_func.system()),
            ChatMessage(role=MessageRole.USER, content=prompt_func.user()),
        ]

        return self.astream_completion(
            messages=messages,
            **kwargs,
        )

    async def astructure_completion_with_prompt(
        self,
        prompt_func: BasePromptFunction,
        response_model: Optional[BaseModel] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Wrapper around astructure_completion to allow for passing in a prompt function.
        """
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt_func.system()),
            ChatMessage(role=MessageRole.USER, content=prompt_func.user()),
        ]

        return await self.astructure_completion(
            messages=messages,
            response_model=response_model,
            **kwargs,
        )

    def astructure_stream_completion_with_prompt(
        self,
        prompt_func: BasePromptFunction,
        response_model: Optional[BaseModel] = None,
        **kwargs: Any,
    ) -> AsyncGenerator:
        """
        Wrapper around astructure_stream_completion to allow for passing in a prompt function.
        """
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt_func.system()),
            ChatMessage(role=MessageRole.USER, content=prompt_func.user()),
        ]

        return self.astructure_stream_completion(
            messages=messages,
            response_model=response_model,
            **kwargs,
        )

    ################################################################################
    #                                                                              #
    #                                    Utils                                     #
    #                                                                              #
    ################################################################################

    async def _aget_chat_completion_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        litellm_kwargs: LiteLLMKwargs = LiteLLMKwargs(),
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        # call the LiteLLM chat completion router with the given messages
        deployment_id = litellm_kwargs.deployment_id
        model = model or deployment_id or self.chat_model
        if model is None:
            raise ValueError("Must provide either a model or a deployment id")

        _, provider, _, _ = get_llm_provider(model)
        if provider == Provider.ollama_chat or provider == Provider.ollama:
            litellm_kwargs.api_key = None  # remove api key for ollama provider
            if (
                litellm_kwargs.frequency_penalty == 0
                or litellm_kwargs.frequency_penalty is None
            ):
                litellm_kwargs.frequency_penalty = 1.1

        try:
            type_hints = get_type_hints(aget_litellm_chat_completion)

            for key, value in litellm_kwargs.model_dump().items():
                if value is not None and key in type_hints and isinstance(value, str):
                    type_hint = type_hints[key]
                    # Handle Optional[Type] annotations
                    if (
                        hasattr(type_hint, "__origin__")
                        and type_hint.__origin__ == Union
                    ):
                        # For Optional types, we assume the first argument is the actual type
                        # (e.g., Optional[int] is represented as Union[int, NoneType])
                        # We cast the value to this type, ignoring the potential None
                        litellm_kwargs[key] = type_hint.__args__[0](value)
                    else:
                        litellm_kwargs[key] = type_hints[key](value)

            completion = await aget_litellm_chat_completion(
                model=model,
                messages=messages,
                **litellm_kwargs.model_dump(exclude_none=True),
            )
            return completion
        except Exception as e:
            if "LLM Provider NOT provided" in str(e):
                logger.error(f"Error: model {model} is not currently supported")
                raise ValueError(f"Model {model} is not currently supported")
            if "'OpenAIError' object has no attribute 'response'" in str(e):
                logger.error("Error: openai key not provided")
                raise ValueError("Openai key not provided")
            logger.error(f"Error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error: {e}")
        except asyncio.CancelledError:
            logger.error("litellm call cancelled")
            raise RuntimeError("litellm call cancelled")

    # TODO: convert to async
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def _get_embeddings_response(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        litellm_kwargs: LiteLLMKwargs = LiteLLMKwargs(),
    ) -> EmbeddingResponse:
        """
        Embed texts using OpenAI's ada model.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, each of which is a list of floats.

        Raises:
            Exception: If the OpenAI API call fails.
        """
        deployment_id = litellm_kwargs.deployment_id
        if deployment_id is not None:
            raise NotImplementedError(
                "Deployment id is currently not supported for embeddings",
            )

        try:
            if litellm_kwargs.base_url is not None:
                # LiteLLM `embedding` has slightly different signature than `completion`
                litellm_kwargs = litellm_kwargs.copy()
                litellm_kwargs.api_base = litellm_kwargs.base_url
                litellm_kwargs.base_url = None

            if model is None:
                model = deployment_id or self.embed_model

            embeddings = get_litellm_embedding(
                model=model,
                input=texts,
                **litellm_kwargs.model_dump(exclude_none=True),
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            raise ServerError.LLM_API_ERROR.as_exception(
                f"LLM API Error, details: {str(e)}",
            )

    def _patch_structure_completion(self):
        original_from_response = OpenAISchema.from_response

        @wraps(original_from_response)
        def patched_from_response(
            cls,
            completion: ChatCompletion,
            validation_context: Optional[dict[str, Any]] = None,
            strict: Optional[bool] = None,
            mode: Mode = Mode.TOOLS,
        ) -> Union[BaseModel, List[BaseModel]]:
            if not hasattr(completion, "choices") or len(completion.choices) <= 1:
                return original_from_response.__func__(
                    cls,
                    completion=completion,
                    validation_context=validation_context,
                    strict=strict,
                    mode=mode,
                )

            results = []
            for choice in completion.choices:
                single_choice = Choice(**choice.model_dump())
                single_completion = ChatCompletion(
                    id=completion.id,
                    choices=[single_choice],
                    created=completion.created,
                    model=completion.model,
                    object=completion.object,
                    usage=completion.usage,
                )
                result = original_from_response.__func__(
                    cls,
                    completion=single_completion,
                    validation_context=validation_context,
                    strict=strict,
                    mode=mode,
                )
                results.append(result)
            return MultiGenerationsResponse(results=results)

        OpenAISchema.from_response = classmethod(patched_from_response)

        # patch the structure completion function
        self._astructure_completion_func = instructor.patch(create=self.acompletion)

    @staticmethod
    def _openai_key():
        return os.getenv("OPENAI_API_KEY")
