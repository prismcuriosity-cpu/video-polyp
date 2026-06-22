"""TensorRT engine building (optional) and a ready-to-run trtexec command.

On the RTX 5090, FP16/BF16 TensorRT typically gives the largest real-time
speed-up.  If the ``tensorrt`` Python bindings are unavailable we still emit the
exact ``trtexec`` command so the engine can be built from the CLI.
"""

from __future__ import annotations


def trtexec_command(onnx_path: str, engine_path: str, fp16: bool = True,
                    workspace_mb: int = 4096, min_bs: int = 1, opt_bs: int = 1,
                    max_bs: int = 4, input_size: int = 512) -> str:
    shape = f"image:{{}}x3x{input_size}x{input_size}"
    return (
        f"trtexec --onnx={onnx_path} --saveEngine={engine_path} "
        f"{'--fp16 ' if fp16 else ''}--memPoolSize=workspace:{workspace_mb} "
        f"--minShapes={shape.format(min_bs)} "
        f"--optShapes={shape.format(opt_bs)} "
        f"--maxShapes={shape.format(max_bs)}"
    )


def build_trt_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                     workspace_mb: int = 4096) -> str:
    """Build a serialized TensorRT engine from an ONNX file.

    Raises a clear, actionable error (with the equivalent trtexec command) if the
    TensorRT bindings are not installed in this environment.
    """
    try:
        import tensorrt as trt  # type: ignore
    except Exception as e:  # pragma: no cover - depends on deployment env
        raise RuntimeError(
            "TensorRT Python bindings not available. Build from the CLI instead:\n  "
            + trtexec_command(onnx_path, engine_path, fp16, workspace_mb)
        ) from e

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse ONNX:\n{errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine)
    return engine_path
