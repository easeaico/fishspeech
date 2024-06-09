import io
import json
import queue
import wave
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from fish_speech.models.vits_decoder.lit_module import VITSDecoder
from fish_speech.models.vqgan.lit_module import VQGAN
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
)
from tools.vits_decoder.inference import load_model

context = {
    "device": "cuda",
    "max_length": 2048,
    "compile": True,
    "precision": torch.bfloat16,
    "ref_json": "ref_data.json",
    "ref_base": "./references",
    "decoder_model": None,
    "llama_queue": None,
    "llama_tokenizer": None,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    await preload_models()
    yield
    context.clear()


app = FastAPI(lifespan=lifespan)


class InvokeRequest(BaseModel):
    text: str = "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游."
    max_new_tokens: int = 0
    chunk_length: Annotated[int, Field(ge=0, le=500, strict=True)] = 150
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.5
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    format: Literal["wav", "mp3", "flac"] = "wav"
    streaming: bool = False


def load_json() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    ref_base = context.get("ref_base")
    json_file = context.get("ref_json")
    if not json_file or not ref_base:
        logger.info("Not using a json file")
        return data

    json_file = f"{ref_base}/{json_file}"
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.warning(f"ref json not found: {json_file}")
    except Exception as e:
        logger.warning(f"Loading json failed: {e}")

    return data


def encode_reference(*, decoder_model, reference_audio, enable_reference_audio):
    if enable_reference_audio and reference_audio is not None:
        # Load audios, and prepare basic info here
        reference_audio_content, _ = librosa.load(
            reference_audio, sr=decoder_model.sampling_rate, mono=True
        )
        audios = torch.from_numpy(reference_audio_content).to(decoder_model.device)[
            None, None, :
        ]
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=decoder_model.device, dtype=torch.long
        )
        logger.info(
            f"Loaded audio with {audios.shape[2] / decoder_model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder
        if isinstance(decoder_model, VQGAN):
            prompt_tokens = decoder_model.encode(audios, audio_lengths)[0][0]
            reference_embedding = None  # VQGAN does not have reference embedding
        elif isinstance(decoder_model, VITSDecoder):
            reference_spec = decoder_model.spec_transform(audios[0])
            reference_embedding = decoder_model.generator.encode_ref(
                reference_spec,
                torch.tensor([reference_spec.shape[-1]], device=decoder_model.device),
            )
            logger.info(f"Loaded reference audio from {reference_audio}")
            prompt_tokens = decoder_model.generator.vq.encode(audios, audio_lengths)[0][
                0
            ]
        else:
            raise ValueError(f"Unknown model type: {type(decoder_model)}")

        logger.info(f"Encoded prompt: {prompt_tokens.shape}")
    elif isinstance(decoder_model, VITSDecoder):
        prompt_tokens = None
        reference_embedding = torch.zeros(
            1, decoder_model.generator.gin_channels, 1, device=decoder_model.device
        )
        logger.info("No reference audio provided, use zero embedding")
    else:
        prompt_tokens = None
        reference_embedding = None
        logger.info("No reference audio provided")

    return prompt_tokens, reference_embedding


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


def decode_vq_tokens(
    *,
    decoder_model,
    codes,
    text_tokens: torch.Tensor,
    reference_embedding: torch.Tensor | None = None,
):
    feature_lengths = torch.tensor([codes.shape[1]], device=decoder_model.device)
    logger.info(f"VQ features: {codes.shape}")

    if isinstance(decoder_model, VQGAN):
        # VQGAN Inference
        return decoder_model.decode(
            indices=codes[None],
            feature_lengths=feature_lengths,
            return_audios=True,
        ).squeeze()

    if isinstance(decoder_model, VITSDecoder):
        # VITS Inference
        quantized = decoder_model.generator.vq.indicies_to_vq_features(
            indices=codes[None], feature_lengths=feature_lengths
        )
        logger.info(f"Restored VQ features: {quantized.shape}")

        return decoder_model.generator.decode(
            quantized,
            torch.tensor([quantized.shape[-1]], device=decoder_model.device),
            text_tokens,
            torch.tensor([text_tokens.shape[-1]], device=decoder_model.device),
            ge=reference_embedding,
        ).squeeze()

    raise ValueError(f"Unknown model type: {type(decoder_model)}")


@torch.inference_mode()
def inference(req: InvokeRequest):
    reference_mapping: dict[str, tuple[str, bytes]] = context["reference_mapping"]
    decoder_model = context["decoder_model"]

    reference_tokens = None
    reference_text = None
    reference_embedding = None

    if req.speaker is not None:
        references = reference_mapping.get(req.speaker)
        if references is not None:
            reference_text, audio_bytes = references

            # Parse reference audio aka prompt
            reference_tokens, reference_embedding = encode_reference(
                decoder_model=decoder_model,
                reference_audio=(io.BytesIO(audio_bytes)),
                enable_reference_audio=True,
            )

    # LLAMA Inference
    request = dict(
        tokenizer=context["llama_tokenizer"],
        device=decoder_model.device,
        max_new_tokens=req.max_new_tokens,
        text=req.text,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        compile=context["compile"],
        iterative_prompt=req.chunk_length > 0,
        chunk_length=req.chunk_length,
        max_length=context["max_length"],
        speaker=req.speaker,
        prompt_tokens=reference_tokens,
        prompt_text=reference_text,
    )

    response_queue = queue.Queue()
    llama_queue = context["llama_queue"]
    llama_queue.put(
        GenerateRequest(
            request=request,
            response_queue=response_queue,
        )
    )

    if req.streaming:
        yield wav_chunk_header()

    llama_tokenizer = context["llama_tokenizer"]
    segments = []
    while True:
        result: WrappedGenerateResponse = response_queue.get()
        if result.status == "error":
            raise result.response

        result: GenerateResponse = result.response
        if result.action == "next":
            break

        text_tokens = llama_tokenizer.encode(result.text, return_tensors="pt").to(
            decoder_model.device
        )

        with torch.autocast(
            device_type=decoder_model.device.type, dtype=context["precision"]
        ):
            fake_audios = decode_vq_tokens(
                decoder_model=decoder_model,
                codes=result.codes,
                text_tokens=text_tokens,
                reference_embedding=reference_embedding,
            )

        fake_audios = fake_audios.float().cpu().numpy()

        if req.streaming:
            yield (fake_audios * 32768).astype(np.int16).tobytes()
        else:
            segments.append(fake_audios)

    if req.streaming:
        return

    if len(segments) == 0:
        raise Exception(
            "No audio generated, please check the input text.",
        )

    fake_audios = np.concatenate(segments, axis=0)
    yield fake_audios


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Literal["wav", "mp3", "flac"] = "wav"


@app.post("/v1/audio/speech")
async def api_infer(req: SpeechRequest):
    """
    Invoke model and generate audio
    """
    generator = inference(
        InvokeRequest(
            text=req.input,
            max_new_tokens=0,
            chunk_length=150,
            top_p=0.7,
            repetition_penalty=1.5,
            temperature=0.7,
            speaker=req.voice,
            emotion=None,
            format=req.response_format,
        )
    )
    fake_audios = next(generator)
    buffer = io.BytesIO()
    decoder_model = context["decoder_model"]
    sf.write(
        buffer, fake_audios, decoder_model.sampling_rate, format=req.response_format
    )

    return StreamingResponse(content=[buffer.getvalue()], media_type="audio/wav")


async def preload_models():
    """
    Preload model
    """
    context["llama_queue"] = launch_thread_safe_queue(
        config_name="dual_ar_2_codebook_medium",
        checkpoint_path="checkpoints/text2semantic-sft-medium.ckpt",
        device="cuda",
        precision=torch.bfloat16,
        max_length=context["max_length"],
        compile=context["compile"],
    )
    context["llama_tokenizer"] = AutoTokenizer.from_pretrained(
        "fishaudio/fish-speech-1"
    )
    logger.info("Llama model loaded, loading vits model...")

    decoder_model = load_model(
        config_name="vits_decoder_finetune",
        checkpoint_path="checkpoints/vits_decoder.ckpt",
        device="cuda",
    )
    context["decoder_model"] = decoder_model

    # Parse reference audio aka prompt
    reference_mapping: dict[str, tuple[str, bytes]] = {}
    context["reference_mapping"] = reference_mapping

    ref_data = load_json()
    ref_base = context["ref_base"]
    for speaker, paths in ref_data.items():
        lab_path, wav_path = paths["ref_lab"], paths["ref_wav"]
        lab_path = f"{ref_base}/{lab_path}"
        wav_path = f"{ref_base}/{wav_path}"

        with open(wav_path, "rb") as wav_file:
            audio_bytes = wav_file.read()
        with open(lab_path, "r", encoding="utf-8") as lab_file:
            ref_text = lab_file.read()

        reference_mapping[speaker] = (ref_text, audio_bytes)

    logger.info("VQ-GAN model loaded, warming up...")

    # Dry run to check if the model
    list(
        inference(
            InvokeRequest(
                text="A warm-up sentence.",
                max_new_tokens=0,
                chunk_length=150,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                speaker=None,
                emotion=None,
                format="wav",
            )
        )
    )


@app.get("/v1/health")
def api_health():
    return {"status": "ok"}
