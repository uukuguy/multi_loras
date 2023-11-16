#!/usr/bin/env python

import os, sys, time
from packaging import version
from pydantic import BaseModel, Field
import uuid
from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast)
from typing import Dict, List, Optional, Tuple, Union, Literal
import multiprocessing as mp
import torch
import socket
import zmq
import zmq.asyncio
import asyncio
import uvloop
import traceback

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template

    _fastchat_available = True
except ImportError:
    _fastchat_available = False

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import uvicorn

from http import HTTPStatus
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse

script_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(script_path)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

GB = 1024 ** 3
MB = 1024 ** 2

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

isFirst = True


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)

@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"

@app.post("/generate")
async def generate(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = []
    count_output_tokens = 0
    tokens = []
    async for request_output, metadata, finished in results_generator:
        count_output_tokens += 1
        if finished == -1:
            return Response(status_code=499)
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await httpserver_manager.abort(request_id)
            return Response(status_code=499)
        final_output.append(request_output)
        if return_details:
            metadata["text"] = request_output
            tokens.append(metadata)

    assert final_output is not None
    ret = {
        "generated_text": ["".join(final_output)],
        "count_output_tokens": count_output_tokens,
    }
    if return_details:
        ret["tokens"] = tokens
    return Response(content=json.dumps(ret, ensure_ascii=False).encode("utf-8"))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata, finished in results_generator:
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": request_output,
                    "logprob": metadata.get("logprob", None),
                    "special": False
                },
                "generated_text": None,
                "finished": finished,
                "details": None
            }

            yield ("data:" + json.dumps(ret, ensure_ascii=False) + f"\n\n").encode(
                "utf-8"
            )

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )

class ChatCompletionRequest(BaseModel):
    # The openai api native parameters
    model: str
    messages: List[Dict[str, str]]
    function_call: Optional[str] = 'none'
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Additional parameters supported by S-LoRA
    do_sample: Optional[bool] = False
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

async def build_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install 'fschat[model_worker,webui]'`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install 'fschat[model_worker,webui]'`")

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")
        # Add a blank message for the assistant. Meaning it's the assistant's turn to talk.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request
) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.n > 1:
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The n parameter currently only supports 1"
        )

    if request.function_call != "none":
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The function call feature is not supported"
        )

    created_time = int(time.time())
    prompt = await build_prompt(request)
    sampling_params = SamplingParams(
        do_sample=request.do_sample,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        max_new_tokens=request.max_tokens,
        stop_sequences=request.stop
    )
    sampling_params.verify()

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    if not request.stream:
        final_output = []
        prompt_tokens = -1
        completion_tokens = 0
        async for request_output, metadata in results_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await httpserver_manager.abort(request_id)
                return Response(status_code=499)
            completion_tokens += 1
            if prompt_tokens == -1:
                prompt_tokens = metadata["prompt_tokens"]
            final_output.append(request_output)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        chat_message = ChatMessage(role="assistant", content="".join(final_output))
        choice = ChatCompletionResponseChoice(index=0, message=chat_message)
        resp = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
        return resp

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata in results_generator:
            delta_message = DeltaMessage(role="assistant", content=request_output)

            stream_choice = ChatCompletionStreamResponseChoice(
                index=0, delta=delta_message
            )

            stream_resp = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + stream_resp.json(ensure_ascii=False) + f"\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )


from slora.sampling_params import SamplingParams
from slora.io_struct import BatchTokenIdOut, ReqDetokenizationState, BatchStrOut, AbortReq, BatchAbortReq

class HttpServerManager:

    def __init__(
        self,
        model_weightdir,
        tokenizor_mode,
        router_port,
        httpserver_port,
        total_token_num,
        max_req_input_len,
        max_req_total_len,
        trust_remote_code,
        dummy=False,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")

        try:
            self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        except:
            if dummy:
                self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode)

        # value type (out_str, metadata, finished, event)
        self.req_id_to_out_inf = {}

        self.total_token_num = total_token_num
        self.max_req_input_len = max_req_input_len
        self.max_req_total_len = max_req_total_len

    async def generate(self, adapter_dir, prompt, sampling_params, request_id):

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)
        if prompt_tokens > self.max_req_input_len:
            raise ValueError(f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}")
        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(
                f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )

        sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        self.send_to_router.send_pyobj((adapter_dir, prompt_ids, sampling_params, request_id))
        event = asyncio.Event()
        self.req_id_to_out_inf[request_id] = ("", {}, False, event)
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            event.clear()
            # request_id is aborted by the backend system for traffic control
            if request_id not in self.req_id_to_out_inf:
                yield "", {}, -1
                break
            out_str, metadata, finished, _ = self.req_id_to_out_inf[request_id]
            if len(metadata) != 0:
                self.req_id_to_out_inf[request_id] = ("", {}, finished, event)
                metadata["prompt_tokens"] = prompt_tokens
                yield out_str, metadata, finished
            if finished:
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                break
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        try:
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def handle_loop(self):
        while True:
            recv_ans: Union(BatchStrOut, BatchAbortReq) = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, (BatchStrOut, BatchAbortReq)), f"error recv type {type(recv_ans)}"
            if isinstance(recv_ans, BatchStrOut):
                for req_id, text, metadata, finished, abort in recv_ans.reqs_infs:
                    try:
                        if not abort:
                            _, _, _, event = self.req_id_to_out_inf[req_id]
                            self.req_id_to_out_inf[req_id] = (
                                text,
                                metadata,
                                finished,
                                event,
                            )
                            event.set()
                        else:
                            del self.req_id_to_out_inf[req_id]
                    except:
                        pass
            elif isinstance(recv_ans, BatchAbortReq):
                print("abort reqs:", recv_ans.reqs)
                for req_id in recv_ans.reqs:
                    try:
                        del self.req_id_to_out_inf[req_id]
                    except:
                        pass

        return


def alloc_can_use_network_port(num=3, used_nccl_port=None):
    port_list = []
    for port in range(10000, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(("localhost", port))
            if result != 0 and port != used_nccl_port:
                port_list.append(port)

            if len(port_list) == num:
                return port_list
    return None


def get_tokenizer(tokenizer_name: str, tokenizer_mode: str = "auto", trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    return tokenizer


def decode_token(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    req: ReqDetokenizationState,
    new_token_id: int,
    skip_special_tokens: bool,
) -> str:

    new_token = tokenizer.convert_ids_to_tokens(new_token_id, skip_special_tokens=skip_special_tokens)

    req.output_tokens.append(new_token)

    if not getattr(tokenizer, "added_tokens_encoder", {}):
        output_text = tokenizer.convert_tokens_to_string(req.output_tokens)
        return output_text

    if skip_special_tokens and new_token in tokenizer.all_special_ids:
        return req.output_str

    if new_token in tokenizer.added_tokens_encoder:
        if req.current_sub_text:
            sub_text = tokenizer.convert_tokens_to_string(req.current_sub_text)
            req.sub_texts.append(sub_text)
            req.current_sub_text = []
        req.sub_texts.append(new_token)
        return " ".join(req.sub_texts)
    else:
        req.current_sub_text.append(new_token)
        new_sub_text = tokenizer.convert_tokens_to_string(req.current_sub_text)
        return " ".join(req.sub_texts + [new_sub_text])


class DeTokenizationManager:

    def __init__(
        self, model_weightdir, tokenizor_mode, detokenization_port, httpserver_port, trust_remote_code, dummy=False
    ):
        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{detokenization_port}")

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://127.0.0.1:{httpserver_port}")

        try:
            self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        except:
            if dummy:
                self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode)

        self.req_id_to_out = {}

    async def handle_loop(self):
        while True:
            try:
                recv_obj: Union(BatchTokenIdOut, ReqDetokenizationState, AbortReq,
                                BatchAbortReq) = await self.recv_from_router.recv_pyobj()
                assert isinstance(
                    recv_obj, (BatchTokenIdOut, ReqDetokenizationState, AbortReq, BatchAbortReq)
                ), f"type is not right {type(recv_obj)}"
                if isinstance(recv_obj, ReqDetokenizationState):
                    self.req_id_to_out[recv_obj.request_id] = recv_obj

                if isinstance(recv_obj, AbortReq):
                    delete_req_id = recv_obj.req_id
                    if delete_req_id in self.req_id_to_out:
                        del self.req_id_to_out[delete_req_id]

                if isinstance(recv_obj, BatchAbortReq):
                    for delete_req_id in recv_obj.reqs:
                        if delete_req_id in self.req_id_to_out:
                            del self.req_id_to_out[delete_req_id]
                    self.send_to_httpserver.send_pyobj(recv_obj)

                if isinstance(recv_obj, BatchTokenIdOut):
                    new_batch_str_out = BatchStrOut()
                    for req_id, new_token_id, new_gen_metadata, finished, abort in recv_obj.reqs_infs:
                        if req_id not in self.req_id_to_out:
                            continue
                        req_out: ReqDetokenizationState = self.req_id_to_out[req_id]
                        req_out.output_ids.append(new_token_id)
                        req_out.gen_metadata.update(new_gen_metadata)
                        out_text = decode_token(self.tokenizer, req_out, new_token_id, skip_special_tokens=True)
                        if out_text.endswith(u'\ufffd'):
                            new_text = ''
                        else:
                            new_text = out_text[len(req_out.output_str):]
                            req_out.output_str = out_text
                        new_batch_str_out.reqs_infs.append(
                            (req_id, new_text, new_gen_metadata, True if abort else finished, abort)
                        )
                        if finished or abort:
                            try:
                                del self.req_id_to_out[req_id]
                            except:
                                pass
                    self.send_to_httpserver.send_pyobj(new_batch_str_out)
            except Exception as e:
                print(f"detoken process has exception {str(e)}")
                traceback.print_exc()
                pass


def start_detokenization_process(args, detokenization_port, httpserver_port, pipe_writer, trust_remote_code=True):
    try:
        router = DeTokenizationManager(
            args.model_dir,
            args.tokenizer_mode,
            detokenization_port=detokenization_port,
            httpserver_port=httpserver_port,
            trust_remote_code=trust_remote_code,
            dummy=args.dummy
        )
    except Exception as e:
        pipe_writer.send(str(e))
        raise
    pipe_writer.send('init ok')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(router.handle_loop())
    return


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    # yapf: disable
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the model weight dir path, the app will load config, weights and tokenizer from this dir")
    parser.add_argument("--tokenizer_mode", type=str, default="slow",
                        help="tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is good for debug and test, when you want to get best performance, try auto mode")
    parser.add_argument("--mode", type=str, default=[], nargs='+',
                        help="Model mode: [int8kv] [int8weight | int4weight]")
    parser.add_argument("--nccl_port", type=int, default=28765,
                        help="the port for nccl to use for communication in distributed training")
    parser.add_argument("--max_total_token_num", type=int, default=6000,
                        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)")
    parser.add_argument("--batch_max_tokens", type=int, default=None,
                        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM")
    parser.add_argument("--eos_id", type=int, default=2, help="eos stop token id")
    parser.add_argument("--running_max_req_size", type=int, default=1000,
                        help="the max size for forward requests in the same time")
    parser.add_argument("--max_req_input_len", type=int, default=2048, help="the max value for req input tokens num")
    parser.add_argument("--max_req_total_len", type=int, default=2048 + 1024,
                        help="the max value for req_input_len + req_output_len")
    parser.add_argument("--tp", type=int, default=1, help="model tp parral size, the default is 1")
    parser.add_argument("--disable_log_stats", action='store_true', help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10, help="log stats interval in second.")

    # ---------- slora arguments ----------
    parser.add_argument("--lora-dirs", type=str, default=[], action="append",
                        help="the adapter weight dirs associate with base model dir")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--pool-size-lora", type=int, default=0)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-size", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="slora")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch-num-adapters", type=int, default=None)
    parser.add_argument("--enable-abort", action="store_true")

    # debug parameters
    # do not use, does not rule out the swap over MemAllocator
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-kernel", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--bmm", action="store_true")
    ''' end of slora arguments '''

    # yapf: enable

    args = parser.parse_args()

    args = parser.parse_args()

    assert args.max_req_input_len < args.max_req_total_len
    setting["max_req_total_len"] = args.max_req_total_len
    setting["nccl_port"] = args.nccl_port

    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens
    else:
        assert (args.batch_max_tokens >= args.max_req_total_len), "batch_max_tokens must >= max_req_total_len"

    return args


# def print_mem_stats(args):
#     model_dir = args.model_dir
#     model_name = args.model_dir.split("/")[-1]
#     try:
#         fake_model = ModelProphet(model_name, model_dir=model_dir)
#     except:
#         fake_model = ModelProphet(model_name)
#     model_size = fake_model.get_model_size()
#     print(f"{model_name}: {model_size / GB:.2f} GB")
#     peak_working_memory = fake_model.get_peak_working_memory(bs=20, context_len=512, tiling_dim=512)
#     print(f"peak working mem for (bs=20, seqlen=512): {peak_working_memory / GB:.2f} GB")
#     peak_working_memory = fake_model.get_peak_working_memory(bs=100, context_len=512, tiling_dim=512)
#     print(f"peak working mem for (bs=100, seqlen=512): {peak_working_memory / GB:.2f} GB")

#     tot_lora_size = 0
#     for lora_dir in args.lora_dirs:
#         lora_name = lora_dir.split("/")[-1]
#         if args.dummy:
#             fake_model = LoRAProphet(lora_name, model_name)
#             try:
#                 fake_model = LoRAProphet(lora_name, model_name)
#             except NotImplementedError as e:
#                 fake_model = LoRAProphet(lora_name, model_name, adapter_dir=lora_dir, base_model_dir=model_dir)
#         else:
#             fake_model = LoRAProphet(lora_name, model_name, adapter_dir=lora_dir, base_model_dir=model_dir)
#         lora_size = fake_model.get_adapter_size()
#         tot_lora_size += lora_size
#         # print(f"{lora_name}, {base_name}: {lora_size / GB:.3f} GB")
#     print(f"all adapters ({len(args.lora_dirs)}) estimated size: {tot_lora_size / GB:.2f} GB")
#     print(f"avg adapter estimated size: {tot_lora_size / len(args.lora_dirs) / MB:.2f} MB")


from slora.common.configs.config import setting
from slora.router.manager import start_router_process

def main():
    args = get_args()

    can_use_ports = alloc_can_use_network_port(num=3 + args.tp, used_nccl_port=args.nccl_port)
    router_port, httpserver_port, detokenization_port = can_use_ports[0:3]
    model_rpc_ports = can_use_ports[3:]

    # -------------------- httpserver_manager --------------------
    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args.model_dir,
        args.tokenizer_mode,
        router_port=router_port,
        httpserver_port=httpserver_port,
        total_token_num=args.max_total_token_num,
        max_req_input_len=args.max_req_input_len,
        max_req_total_len=args.max_req_total_len,
        trust_remote_code=True,
    )
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)
    proc_router = mp.Process(
        target=start_router_process,
        args=(
            args,
            router_port,
            detokenization_port,
            model_rpc_ports,
            args.mode,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenization_process,
        args=(
            args,
            detokenization_port,
            httpserver_port,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()

    # wait load model ready
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print(
            "router init state:",
            router_init_state,
            "detoken init state:",
            detoken_init_state,
        )
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    # print_mem_stats(args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )

if __name__ == "__main__":
    # this code will not be ok for settings to fork to subprocess
    torch.multiprocessing.set_start_method('spawn'),
    main()
