"""Driver Script for the ASPLOS OPTNKI Contest"""
import argparse
import ast
import base64
import copy
import json
import os
import time

from torch_neuronx.pyhlo.hlo_pb2 import HloModuleProto
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig, to_torch_dtype
from neuronx_distributed_inference.utils.accuracy import (
    get_generate_outputs,
    check_accuracy_logits
)
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed

# Load the model for ASPLOS contest
from llama import NeuronLlamaForCausalLM

set_random_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()

    # ASPLOS contest specific
    parser.add_argument("--mode", choices=["evaluate", "validate", "generate"])
    parser.add_argument("--enable-nki", action="store_true")

    # Model path
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/models/llama-3.2-1b/")
    parser.add_argument("--compiled-model-path", type=str,
                        default="/home/ubuntu/traced_model/llama-3.2-1b/")

    # Evaluation
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--divergence-difference-tol", type=float, default=0.001)
    parser.add_argument("--tol-map", type=str)
    parser.add_argument("--num-tokens-to-check", type=int)

    # Generation
    parser.add_argument("--prompt", dest="prompts", type=str, action="append",
                        default="I believe the meaning of life is")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--global-topk", type=int)
    parser.add_argument("--do-sample", type=bool, default=True)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--pad-token-id", type=int, default=2)

    # Basic config
    parser.add_argument("--torch-dtype", type=to_torch_dtype, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--padding-side", type=str)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-active-tokens", type=int)
    parser.add_argument("--n-positions", type=int)
    parser.add_argument("--max-context-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    parser.add_argument("--output-logits", action="store_true")
    parser.add_argument("--vocab-parallel", action="store_true")

    # Attention
    parser.add_argument("--fused-qkv", action="store_true")
    parser.add_argument("--sequence-parallel-enabled", action="store_true")
    parser.add_argument("--flash-decoding-enabled", action="store_true")

    # On device sampling
    parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    parser.add_argument("--enable-bucketing", type=bool, default=True)
    parser.add_argument("--bucket-n-active-tokens", action="store_true")
    parser.add_argument("--context-encoding-buckets", nargs="+", type=int)
    parser.add_argument("--token-generation-buckets", nargs="+", type=int)

    # Parallelism
    parser.add_argument("--tp-degree", type=int, default=2)

    # Kernels
    parser.add_argument("--qkv-kernel-enabled", action="store_true")
    parser.add_argument("--attn-kernel-enabled", action="store_true")
    parser.add_argument("--mlp-kernel-enabled", action="store_true")
    parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    parser.add_argument("--quantized-kernel-lower-bound", type=float, default=1200.0)
    parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")

    return parser.parse_args()


def validate_file_exists(path):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise argparse.ArgumentError("Path must exist and be a file")
    return path


def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def prepare_inference(model_cls, args):
    # Initialize configs.
    print("Loading configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    if args.on_device_sampling:
        config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)

    model_cls = NeuronLlamaForCausalLM
    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)

    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(args.model_path)
    )

    model = model_cls(args.model_path, config)

    # Compile and save model.
    compiling_start_time = time.monotonic()
    print("\nCompiling and saving model...")
    model.compile(args.compiled_model_path, debug=False)

    compiling_end_time = time.monotonic()
    total_compiling_time = compiling_end_time - compiling_start_time
    print(f"Compiling and tracing time: {total_compiling_time} seconds")

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(args.compiled_model_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - compiling_end_time
    print(f"Total model loading time: {model_loading_time} seconds")

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = [
        "do_sample",
        "top_k",
        "pad_token_id",
        "dynamic",
        "top_p",
        "temperature",
    ]
    generation_config_kwargs = {
        k: getattr(args, k) for k in generation_config_args if getattr(args, k) is not None
    }
    generation_config.update(**generation_config_kwargs)

    return model, tokenizer, generation_config


def run_generation(model, tokenizer, prompts, generation_config):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")

    _, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        generation_config=generation_config,
        max_length=model.neuron_config.max_length,
    )

    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


def run_accuracy_check(
    model,
    tokenizer,
    generation_config,
    prompt,
    divergence_difference_tol,
    tol_map,
    num_tokens_to_check=None,
):
    if tol_map:
        tol_map = ast.literal_eval(tol_map)

    try:
        check_accuracy_logits(
            model,
            tokenizer,
            generation_config,
            prompt=prompt,
            expected_logits=None,
            divergence_difference_tol=divergence_difference_tol,
            tol_map=tol_map,
            num_tokens_to_check=num_tokens_to_check,
        )
    except AssertionError:
        return False

    return True


def count_nki_flop_ratio(
    hlo_path_context_enc="/tmp/nxd_model/context_encoding_model/_tp0_bk0/model/graph.hlo",
    hlo_path_token_gen="/tmp/nxd_model/token_generation_model/_tp0_bk0/model/graph.hlo"
):
    hlo_macs = 0
    nki_macs = 0

    def parse_hlo_file(hlo_file_path):
        with open(hlo_file_path, 'rb') as f:
            hlo_data = f.read()

        hlo_proto = HloModuleProto()
        hlo_proto.ParseFromString(hlo_data)
        return hlo_proto

    def count_mac(hlo_proto):
        nki_mac = 0
        hlo_mac = 0

        for computation in hlo_proto.computations:
            instruction_map = {instr.id: instr for instr in computation.instructions}

            for instruction in computation.instructions:
                # Finding NKI ops
                if instruction.opcode == "custom-call":
                    if instruction.custom_call_target == 'AwsNeuronCustomNativeKernel':
                        try:
                            backend_config = instruction.backend_config
                            config = json.loads(base64.b64decode(backend_config))
                            mac_count = int(config['mac_count'])
                        except Exception:
                            mac_count = 0

                        nki_mac += mac_count
                        hlo_mac += mac_count
                elif instruction.opcode == "dot":
                    # Get dot dimension numbers
                    dnums = instruction.dot_dimension_numbers

                    # Get shapes of operands using operand_ids
                    lhs_shape = instruction_map[instruction.operand_ids[0]].shape
                    rhs_shape = instruction_map[instruction.operand_ids[1]].shape

                    # Initialize counters
                    lhs_batch = 1
                    lhs_contracting_size = 1
                    lhs_non_contracting_size = 1
                    rhs_non_contracting_size = 1

                    # Process LHS shape
                    for i in range(len(lhs_shape.dimensions)):
                        if i in dnums.lhs_contracting_dimensions:
                            lhs_contracting_size *= lhs_shape.dimensions[i]
                        elif i in dnums.lhs_batch_dimensions:
                            lhs_batch *= lhs_shape.dimensions[i]
                        else:
                            lhs_non_contracting_size *= lhs_shape.dimensions[i]

                    # Process RHS shape
                    for i in range(len(rhs_shape.dimensions)):
                        if i not in dnums.rhs_contracting_dimensions and \
                           i not in dnums.rhs_batch_dimensions:
                            rhs_non_contracting_size *= rhs_shape.dimensions[i]

                    mac_count = (lhs_batch * lhs_non_contracting_size *
                                 lhs_contracting_size * rhs_non_contracting_size)
                    hlo_mac += mac_count

        return hlo_mac, nki_mac

    hlo_proto_context_enc = parse_hlo_file(hlo_path_context_enc)
    hlo_proto_token_gen = parse_hlo_file(hlo_path_token_gen)
    hlo_mac_context_enc, nki_mac_context_enc = count_mac(hlo_proto_context_enc)
    hlo_mac_token_gen, nki_mac_token_gen = count_mac(hlo_proto_token_gen)

    # FIXME: Need to consider token gen get executed more
    hlo_macs = hlo_mac_context_enc + hlo_mac_token_gen
    nki_macs = nki_mac_context_enc + nki_mac_token_gen

    if hlo_macs == 0:
        assert nki_macs == 0
        nki_flop_ratio = 0
    else:
        nki_flop_ratio = nki_macs / hlo_macs

    return nki_flop_ratio


def calculate_score(accuracy, latency, throughput, nki_flop_ratio):

    # latency and throughput
    LATENCY_BASE = 295.20
    THROUGHPUT_BASE = 217.95

    increased_throughput = throughput / THROUGHPUT_BASE
    reduced_latency = LATENCY_BASE / latency

    # FIXME: adjust this calculation function
    final_score = accuracy * reduced_latency * increased_throughput * (1 + nki_flop_ratio)

    return final_score


def main():
    args = parse_args()
    model, tokenizer, generation_config = prepare_inference(NeuronLlamaForCausalLM, args)

    if args.mode == "generate":
        run_generation(
            model,
            tokenizer,
            args.prompts,
            generation_config
        )

    elif args.mode == "validate":

        passed = run_accuracy_check(
            model,
            tokenizer,
            generation_config,
            args.prompts[0],
            args.divergence_difference_tol,
            args.tol_map,
            num_tokens_to_check=args.num_tokens_to_check,
        )

        status = "passed" if passed else "failed"
        print(f"Validation {status}.")

    elif args.mode == "evaluate":

        accuracy = run_accuracy_check(
            model,
            tokenizer,
            generation_config,
            args.prompts[0],
            args.divergence_difference_tol,
            args.tol_map,
            num_tokens_to_check=args.num_tokens_to_check,
        )

        report = benchmark_sampling(model, None, generation_config)

        latency = report["e2e_model"]["latency_ms_p99"]
        throughput = report["e2e_model"]["throughput"]

        nki_flop_ratio = count_nki_flop_ratio()

        score = calculate_score(accuracy, latency, throughput, nki_flop_ratio)
        print(
            f"Final Score: {score}\n"
            f"\tAccuracy: {accuracy}\n"
            f"\tLatency: {latency}\n"
            f"\tThroughput: {throughput}\n"
            f"\tNKI FLOPs Ratio: {nki_flop_ratio}"
        )

    else:
        assert False, "Undefined mode"


if __name__ == "__main__":
    main()
