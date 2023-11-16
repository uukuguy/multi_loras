#!/usr/bin/env python
from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast)
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp
import torch
import socket
import zmq
import zmq.asyncio
import asyncio
import uvloop
import traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse

GB = 1024 ** 3
MB = 1024 ** 2

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

isFirst = True


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)



class SamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,  # -1 is for all
        ignore_eos: bool = False,
        max_new_tokens: int = 16,
        stop_sequences: Optional[Union[str, List[str]]] = None  # 停止句子条件
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.ignore_eos = ignore_eos
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = stop_sequences
        if self.do_sample == False:
            self.temperature = 1.0
            self.top_p = 1.0
            self.top_k = 1
        # temperature is too slow, change to greedy search
        if self.temperature >= 0.0 and self.temperature < 1e-5:
            self.temperature = 1.0
            self.top_k = 1
        return

    def verify(self):
        if self.presence_penalty < 0.0:
            raise ValueError(f"presence_penalty must >= 0.0, got {self.presence_penalty}")
        if self.frequency_penalty < 0.0:
            raise ValueError(f"frequency_penalty must >= 0.0, got {self.frequency_penalty}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must > 0.0, got {self.temperature}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must in (0.0, 1.0], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, got {self.top_k}.")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1 , got {self.max_new_tokens}.")
        return

    def stop_sentences_to_token_ids(self, tokenizer):
        if self.stop_sequences is None:
            self.stop_sequences = []
        else:
            if isinstance(self.stop_sequences, str):
                self.stop_sequences = [self.stop_sequences]
            new_stop_sequences = []
            for stop_str in self.stop_sequences:
                stop_str_ids = tokenizer.encode(stop_str)
                # remove bos_token_id
                if stop_str_ids is not None and len(stop_str_ids) >= 1:
                    stop_str_ids = stop_str_ids[1:]
                if len(stop_str_ids) > 0:
                    new_stop_sequences.append(stop_str_ids)
            self.stop_sequences = new_stop_sequences
        return

    def to_dict(self):
        ret = {}
        ret["do_sample"] = self.do_sample
        ret["presence_penalty"] = self.presence_penalty
        ret["frequency_penalty"] = self.frequency_penalty
        ret["temperature"] = self.temperature
        ret["top_p"] = self.top_p
        ret["top_k"] = self.top_k
        # if self.ignore_eos is not None:
        #     ret["ignore_eos"] = self.ignore_eos
        # if self.max_tokens is not None:
        #     ret["max_tokens"] = self.max_tokens
        return ret


class Req:

    def __init__(self, adapter_dir, request_id, prompt_ids, sample_params: SamplingParams):
        self.adapter_dir = adapter_dir
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False

    def to_rpc_obj(self):
        return {
            "adapter_dir": self.adapter_dir,
            "request_id": self.request_id,
            "input_id": self.prompt_ids,
            "output_len": self.max_output_len,
            "sampling_param": self.sample_params.to_dict()
        }

    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(
            self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos
        )
        if self.output_metadata_list:
            out.gen_metadata.update(self.output_metadata_list[-1])
        return out

    def stop_sequences_matched(self):
        for stop_token_ids in self.sample_params.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if len(self.output_ids) >= stop_len:
                    if all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len)):
                        return True
        return False

    def __repr__(self):
        return (
            f"request_id(n={self.request_id}, "
            f"adapter_dir={self.adapter_dir}, "
            f"prompt_ids={self.prompt_ids}, "
        )


class ReqDetokenizationState:

    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}


class Batch:

    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req
                           for req in reqs}

        self.adapter_dirs = set()
        for req in reqs:
            self.adapter_dirs.add(req.adapter_dir)

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len
        return tokens

    def calcu_used_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + len(req.output_ids)
        return tokens

    def mark_finished_req(self, eos_id):
        has_new_finish = False
        for req in self.reqs:
            if req.stop_sequences_matched():
                req.has_generate_finished = True
                has_new_finish = True
            if req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                req.has_generate_finished = True
                has_new_finish = True
            if len(req.output_ids) >= req.max_output_len or req.aborted:
                req.has_generate_finished = True
                has_new_finish = True
        return has_new_finish

    def filter_finished(self):
        unfinished_req = []
        for req in self.reqs:
            if not req.has_generate_finished:
                unfinished_req.append(req)
        self.reqs = unfinished_req
        self.id_to_reqs = {req.request_id: req
                           for req in self.reqs}

        self.adapter_dirs = set()
        for req in self.reqs:
            self.adapter_dirs.add(req.adapter_dir)

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
            self.adapter_dirs.add(_req.adapter_dir)
        self.id_to_reqs = {req.request_id: req
                           for req in self.reqs}
        return

    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                # f"reqs={self.reqs}, "
                f"req_ids={self.id_to_reqs.keys()}")


class BatchTokenIdOut:

    def __init__(self):
        # [req_id, new_token_id, gen_metadata, finished_state, abort_state]
        self.reqs_infs: List[Tuple[str, int, Dict, bool, bool]] = []


class BatchStrOut:

    def __init__(self):
        # [req_id, token_str, gen_metadata, finished_state, abort_state]
        self.reqs_infs: List[Tuple[str, str, Dict, bool, bool]] = []


class AbortReq:

    def __init__(self, req_id):
        self.req_id = req_id


class BatchAbortReq:

    def __init__(self, req_ids):
        self.reqs: List[str] = req_ids


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


def start_detokenization_process(args, detokenization_port, httpserver_port, pipe_writer, trust_remote_code):
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
    parser.add_argument("--max_req_input_len", type=int, default=2048, help="the max value for req input tokens num")
    parser.add_argument("--max_req_total_len", type=int, default=2048 + 1024,
                        help="the max value for req_input_len + req_output_len")
    parser.add_argument("--tp", type=int, default=1, help="model tp parral size, the default is 1")

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
    # yapf: enable

    args = parser.parse_args()
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


from .router.manager import start_router_process


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
            args.trust_remote_code,
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
