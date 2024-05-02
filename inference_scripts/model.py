import os
from enum import Enum
from threading import Thread
from inference_scripts.rag import RAG_INSTANCE
from typing import Any, Iterator, Union, List
from huggingface_hub import snapshot_download


class INFERENCE:
    def __init__(
        self,
        model_path: str = "",
        backend_type: str = "transformers",
        max_tokens: int = 4000,
        load_in_8bit: bool = True,
        verbose: bool = False,
    ):
        """Load a transformer model from `model_path`.

        Args:
            model_path: Path to the model.
            backend_type: Backend for model, options: llama.cpp, gptq, transformers, transformers so far only
            max_tokens: Maximum context size.
            load_in_8bit: Use bitsandbytes to run model in 8 bit mode (only for transformers models).
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A INFERENCE instance.
        """
        self.model_path = model_path
        self.backend_type = BackendType.get_type(backend_type)
        self.max_tokens = max_tokens
        self.load_in_8bit = load_in_8bit

        self.model = None
        self.tokenizer = None

        self.verbose = verbose

        import torch

        if torch.cuda.is_available():
            print("Running on GPU with backend torch transformers.")
        else:
            print("GPU CUDA not found.")
            
        self.mistral7b = "./models/mistral7b"
        # download mistral7b if model path empty 
        if self.model_path == "":
            print("Model path is empty.")
            if not os.path.exists(self.mistral7b):
                print("Start downloading model to: " + self.mistral7b)
                snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.2", local_dir=self.mistral7b)
            else:
                print("Model exists in " + self.mistral7b)
            self.model_path = self.mistral7b
            

        self.init_tokenizer()
        self.init_model()

    def init_model(self):
        if self.model is None:
            self.model = INFERENCE.create_model(
                self.model_path,
                self.backend_type,
                self.max_tokens,
                self.load_in_8bit,
                self.verbose,
            )
        if self.backend_type is not BackendType.LLAMA_CPP:
            self.model.eval()

    def init_tokenizer(self):
        if self.backend_type is not BackendType.LLAMA_CPP:
            if self.tokenizer is None:
                self.tokenizer = INFERENCE.create_tokenizer(self.model_path)

    @classmethod
    def create_model(
        cls, model_path, backend_type, max_tokens, load_in_8bit, verbose
    ):
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=load_in_8bit,
        )
        return model

    @classmethod
    def create_tokenizer(cls, model_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def get_token_length(
        self,
        prompt: str,
    ) -> int:
        if self.backend_type is BackendType.LLAMA_CPP:
            input_ids = self.model.tokenize(bytes(prompt, "utf-8"))
            return len(input_ids)
        else:
            print(prompt)
            input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
            return input_ids.shape[-1]

    def get_input_token_length(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
    ) -> int:
        prompt = self.get_prompt(message=message, chat_history=chat_history, system_prompt=system_prompt)
        return self.get_token_length(prompt)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Create a generator of response from a prompt.

        Examples:
            >>> model_instance = INFERENCE()
            >>> prompt = get_prompt("Hi do you know Pytorch?")
            >>> for response in model_instance.generate(prompt):
            ...     print(response)

        Args:
            prompt: The prompt to generate text from.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Yields:
            The generated text.
        """
        if self.backend_type is BackendType.LLAMA_CPP:
            result = self.model(
                prompt=prompt,
                stream=True,
                max_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            outputs = []
            for part in result:
                text = part["choices"][0]["text"]
                outputs.append(text)
                yield "".join(outputs)
        else:
            from transformers import TextIteratorStreamer

            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # num_beams=1,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()

            outputs = []
            for text in streamer:
                outputs.append(text)
                yield "".join(outputs)

    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        max_new_tokens: int = 1000,
        temperature: float = 0.01,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        """Create a generator of response from a chat message.
        Process message to llama2 prompt with chat history
        and system_prompt for chatbot.

        Args:
            message: The origianl chat message to generate text from.
            chat_history: Chat history list from chatbot.
            system_prompt: System prompt for chatbot.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Yields:
            The generated text.
        """
        prompt = self.get_prompt(message, chat_history, system_prompt)
        return self.generate(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

    def __call__(
        self,
        prompt: str,
        stream: bool = False,
        max_new_tokens: int = 1000,
        temperature: float = 0.01,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Generate text from a prompt.

        Examples:
            >>> model_instance = INFERENCE()
            >>> prompt = get_prompt("Hi do you know Pytorch?")
            >>> print(model_instance(prompt))

        Args:
            prompt: The prompt to generate text from.
            stream: Whether to stream the results.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Generated text.
        """
        if self.backend_type is BackendType.LLAMA_CPP:
            completion_or_chunks = self.model.__call__(
                prompt,
                stream=stream,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            if stream:

                def chunk_generator(chunks):
                    for part in chunks:
                        chunk = part["choices"][0]["text"]
                        yield chunk

                chunks: Iterator[str] = chunk_generator(completion_or_chunks)
                return chunks
            return completion_or_chunks["choices"][0]["text"]
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids
            prompt_tokens_len = len(inputs[0])
            inputs = inputs.to("cuda")
            generate_kwargs = dict(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # num_beams=1,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            if stream:
                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=10.0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generate_kwargs["streamer"] = streamer

                t = Thread(target=self.model.generate, kwargs=generate_kwargs)
                t.start()
                return streamer
            else:
                output_ids = self.model.generate(
                    **generate_kwargs,
                )
                # skip prompt, skip special tokens
                output = self.tokenizer.decode(
                    output_ids[0][prompt_tokens_len:], skip_special_tokens=True
                )
                return output


    from inference_scripts.rag import RAG_INSTANCE
    context_rag = RAG_INSTANCE()

    # modified to use tokenizer chat template
    def get_prompt(self, message: str, chat_history: list[tuple[str, str]] = [], system_prompt: str = ""
        ) -> str:
        """Process message to final prompt with chat history
        and system_prompt for chatbot.

        Examples:
            >>> prompt = get_prompt("Hi do you know Pytorch?")

        Args:
            message: The origianl chat message to generate text from.
            chat_history: Chat history list from chatbot.
            system_prompt: System prompt for chatbot.

        Yields:
            prompt string.
        """ 
        # print("pr"+message)
        final_input_rag = self.context_rag.query(message, chat_history)
        # print(final_input_rag)
        template_sequence = []
        # if system_prompt != "":         # add system prompt
            # template_sequence.append({"role": "system", "content": system_prompt}) # add system prompt
        if len(chat_history) > 0:      # add chat history, chat_history only contains past chat
            for i in range(len(chat_history)):
                user_input, response = chat_history[i]
                template_sequence.append({"role": "user", "content": user_input})
                template_sequence.append({"role": "assistant", "content": response})
        template_sequence.append({"role": "user", "content": final_input_rag}) # add user input
        final_full_prompt = self.tokenizer.apply_chat_template(template_sequence, add_generation_prompt=True, tokenize = False)
        return final_full_prompt


class BackendType(Enum):
    UNKNOWN = 0
    TRANSFORMERS = 1
    GPTQ = 2
    LLAMA_CPP = 3

    @classmethod
    def get_type(cls, backend_name: str):
        backend_type = None
        backend_name_lower = backend_name.lower()
        if "transformers" in backend_name_lower:
            backend_type = BackendType.TRANSFORMERS
        elif "gptq" in backend_name_lower:
            backend_type = BackendType.GPTQ
        elif "cpp" in backend_name_lower:
            backend_type = BackendType.LLAMA_CPP
        else:
            raise Exception("Unknown backend: " + backend_name)
            # backend_type = BackendType.UNKNOWN
        return backend_type
