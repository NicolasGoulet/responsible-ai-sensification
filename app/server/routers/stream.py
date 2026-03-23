"""stream.py: WebSocket endpoint for live pipeline streaming."""
import asyncio
import dataclasses
import json
import logging
import math
import sys

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.server.session import PipelineParams, PipelineSession  # noqa: F401 (PipelineSession used for _session)

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level model cache: (model_id, layer, width) -> {"model": ..., "tokenizer": ..., "sae": ..., "neuronpedia": ...}
_model_cache: dict[tuple, dict] = {}

# Cluster map cache: (model_id, layer, width, clusters) -> cluster_map dict
_cluster_cache: dict[tuple, dict] = {}

# One shared session (single-user for now)
_session = PipelineSession()


async def _send(ws: WebSocket, msg: dict) -> None:
    await ws.send_text(json.dumps(msg))


def _midi_to_frequency(midi_note: int) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return 440.0 * math.pow(2.0, (midi_note - 69) / 12.0)


def _apply_browser_pitch_policy(
    notes: list[dict],
    active_features: list[dict],
    tonal_context,
    *,
    token: str,
    token_id: int,
    l0: int,
    elapsed_ms: int,
):
    """Mutate browser notes in place using the prompt-level tonal context.

    This keeps the browser path changes localized in stream.py for now. The CLI path is
    intentionally left unchanged until we decide how to pass prompt-level tonal context
    through the CLI pipeline.
    """
    if tonal_context is None:
        return notes

    from app.server.pipeline.pitch_policy import TokenPitchInput, choose_pitch

    for feature, note in zip(active_features, notes, strict=False):
        decision = choose_pitch(
            tonal_context,
            TokenPitchInput(
                feature_index=feature["index"],
                activation=feature["activation"],
                token_id=token_id,
                token=token,
                cluster=note.get("cluster"),
                instrument=note.get("instrument"),
                l0=l0,
                elapsed_ms=elapsed_ms,
            ),
        )
        note["freq"] = _midi_to_frequency(decision.chosen_midi)
        note["pitch_midi"] = decision.chosen_midi
        note["raw_pitch_midi"] = decision.raw_midi
        note["used_scale_bias"] = decision.used_scale_bias
        note["key_name"] = decision.key_name
    return notes


async def _run_pipeline(ws: WebSocket, params: PipelineParams) -> None:
    """Background task: load model (cached), run inspect_live, stream token events."""
    try:
        cache_key = (params.model, params.layer, params.width)

        if cache_key in _model_cache:
            print("[pipeline] Using cached model/SAE/neuronpedia.", file=sys.stderr, flush=True)
            await _send(ws, {"type": "loading", "stage": "cached"})
            cached = _model_cache[cache_key]
            model = cached["model"]
            tokenizer = cached["tokenizer"]
            sae = cached["sae"]
            neuronpedia = cached["neuronpedia"]
        else:
            # Load model
            print(f"[pipeline] Loading language model: {params.model}", file=sys.stderr, flush=True)
            await _send(ws, {"type": "loading", "stage": "Loading language model..."})

            def _load_model():
                from transformers import AutoModelForCausalLM, AutoTokenizer
                _model = AutoModelForCausalLM.from_pretrained(params.model, device_map="auto")
                _tokenizer = AutoTokenizer.from_pretrained(params.model)
                return _model, _tokenizer

            model, tokenizer = await asyncio.to_thread(_load_model)
            print("[pipeline] Language model loaded.", file=sys.stderr, flush=True)

            # Load SAE
            print(f"[pipeline] Loading SAE (layer={params.layer}, width={params.width}, l0={params.l0})", file=sys.stderr, flush=True)
            await _send(ws, {"type": "loading", "stage": f"Loading SAE (layer={params.layer}, width={params.width})..."})

            def _load_sae():
                from app.server.pipeline.extract import load_sae
                return load_sae(layer=params.layer, width=params.width, l0=params.l0)

            sae = await asyncio.to_thread(_load_sae)
            print("[pipeline] SAE loaded.", file=sys.stderr, flush=True)

            # Load Neuronpedia
            print("[pipeline] Loading Neuronpedia explanations...", file=sys.stderr, flush=True)
            await _send(ws, {"type": "loading", "stage": "Downloading Neuronpedia explanations..."})

            def _load_neuronpedia():
                from app.server.pipeline.extract import download_neuronpedia_explanations
                # Neuronpedia uses "gemma-3-1b" style key, strip the HF org prefix
                np_model_id = params.model.split("/")[-1].replace("-pt", "")
                return download_neuronpedia_explanations(np_model_id, params.layer, params.width)

            neuronpedia = await asyncio.to_thread(_load_neuronpedia)
            print(f"[pipeline] Neuronpedia loaded: {len(neuronpedia.explanations)} features.", file=sys.stderr, flush=True)

            _model_cache[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "sae": sae,
                "neuronpedia": neuronpedia,
            }

        tonal_context = None
        try:
            from app.server.pipeline.tonality import DEFAULT_CACHE_DIR
            from app.server.pipeline.tonality_matcher import match_prompt_to_tonalities
            from app.server.pipeline.pitch_policy import TonalityContext

            cache_path = DEFAULT_CACHE_DIR / "schubart_default_all-MiniLM-L6-v2.json"
            if cache_path.exists():
                await _send(ws, {"type": "loading", "stage": "Choosing tonality..."})
                tonality_result = await asyncio.to_thread(
                    match_prompt_to_tonalities,
                    params.prompt,
                    cache_path,
                    3,
                    None,
                )
                tonal_context = TonalityContext(matches=tonality_result.matches)
                tonality_summary = ", ".join(
                    f"{match.key} ({match.score:.3f})" for match in tonality_result.matches
                )
                print(f"[pipeline] Tonality matches: {tonality_summary}", file=sys.stderr, flush=True)
                logger.info("Tonality matches selected: %s", tonality_summary)
                await _send(
                    ws,
                    {
                        "type": "tonality",
                        "matches": [dataclasses.asdict(match) for match in tonality_result.matches],
                    },
                )
            else:
                logger.info("Tonality cache missing at %s; browser tonal matching skipped", cache_path)
        except Exception:
            logger.exception("Tonality matching failed; continuing without browser tonal bias")

        await _send(ws, {"type": "loading", "stage": "Running generation..."})

        from app.server.pipeline.extract import inspect_live
        from app.server.pipeline.transform import apply_identity, apply_cluster, build_cluster_map

        cluster_map: dict = {}
        if params.strategy == "cluster":
            cluster_key = (params.model, params.layer, params.width, params.clusters)
            if cluster_key in _cluster_cache:
                cluster_map = _cluster_cache[cluster_key]
                print(f"[pipeline] Cluster map from in-memory cache: {len(cluster_map)} entries.", file=sys.stderr, flush=True)
                logger.info("Cluster map loaded from in-memory cache: %d entries", len(cluster_map))
            else:
                print(f"[pipeline] Building cluster map (clusters={params.clusters})...", file=sys.stderr, flush=True)
                await _send(ws, {"type": "loading", "stage": "Building cluster map..."})

                def _build_clusters():
                    np_model_id = params.model.split("/")[-1].replace("-pt", "")
                    return build_cluster_map(
                        np_model_id,
                        params.layer,
                        params.width,
                        params.clusters,
                        "all-MiniLM-L6-v2",
                    )

                cluster_map = await asyncio.to_thread(_build_clusters)
                _cluster_cache[cluster_key] = cluster_map

            print(f"[pipeline] Cluster map ready: {len(cluster_map)} entries.", file=sys.stderr, flush=True)
            logger.info("Cluster map ready: %d entries", len(cluster_map))

        print(f"[pipeline] Starting pipeline: prompt={params.prompt!r} strategy={params.strategy} clusters={params.clusters}", file=sys.stderr, flush=True)
        logger.info("Starting pipeline: prompt=%r strategy=%s clusters=%s", params.prompt, params.strategy, params.clusters)

        # Producer + synthesizer: run generation concurrently with timed playback
        queue: asyncio.Queue = asyncio.Queue()
        collected: list[dict] = []
        event_loop = asyncio.get_running_loop()

        async def _producer():
            try:
                token_count = 0

                def _generate():
                    nonlocal token_count
                    print(f"[pipeline] Generation starting (max_tokens={params.max_tokens})...", file=sys.stderr, flush=True)
                    logger.info("Starting generation: max_tokens=%d", params.max_tokens)
                    for token_analysis, elapsed_ms in inspect_live(
                        params.prompt, model, tokenizer, sae,
                        params.layer, neuronpedia,
                        max_new_tokens=params.max_tokens,
                    ):
                        token_count += 1
                        active_features = [f.model_dump() for f in token_analysis.active_features]
                        if params.strategy == "cluster":
                            notes = apply_cluster(active_features, cluster_map)
                        else:
                            notes = apply_identity(active_features)

                        notes = _apply_browser_pitch_policy(
                            notes,
                            active_features,
                            tonal_context,
                            token=token_analysis.token,
                            token_id=token_analysis.token_id,
                            l0=token_analysis.l0,
                            elapsed_ms=elapsed_ms,
                        )

                        event = {
                            "type": "token",
                            "token": token_analysis.token,
                            "token_id": token_analysis.token_id,
                            "elapsed_ms": elapsed_ms,
                            "notes": notes,
                        }
                        logger.debug("Token %d generated in %dms: %r", token_count, elapsed_ms, token_analysis.token)
                        asyncio.run_coroutine_threadsafe(queue.put(event), event_loop).result()

                await asyncio.to_thread(_generate)
                print(f"[pipeline] Generation complete: {token_count} tokens.", file=sys.stderr, flush=True)
                logger.info("Generation complete: %d tokens", token_count)
            finally:
                await queue.put(None)  # always signal done, even on error

        async def _synthesizer():
            # Initial playback: drain the queue
            while True:
                event = await queue.get()
                if event is None:
                    break
                collected.append(event)
                await _send(ws, event)
                if params.mode == "timed":
                    await asyncio.sleep(60.0 / params.bpm)

            await _send(ws, {"type": "done"})

            # Post-generation: loop or idle until cancelled
            loop_count = 0
            was_looping = False
            while True:
                if params.loop:
                    was_looping = True
                    loop_count += 1
                    for event in collected:
                        if not params.loop:
                            break
                        await _send(ws, {**event, "loop_count": loop_count})
                        if params.mode == "timed":
                            await asyncio.sleep(60.0 / params.bpm)
                else:
                    if was_looping:
                        was_looping = False
                        await _send(ws, {"type": "silent"})
                    await asyncio.sleep(0.1)

        producer_task = asyncio.create_task(_producer())
        await _synthesizer()
        await producer_task  # re-raises any exception from generation

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Pipeline error")
        try:
            await _send(ws, {"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await _send(ws, {"type": "stopped"})
        except Exception:
            pass


@router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    await _send(ws, {"type": "ready", "params": dataclasses.asdict(_session.params)})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(ws, {"type": "error", "message": "Invalid JSON"})
                continue

            action = msg.get("action")

            if action == "start":
                await _session.cancel()
                raw_params = msg.get("params", {})
                _session.params = PipelineParams()
                _session.params.update(**raw_params)
                _session.task = asyncio.create_task(
                    _run_pipeline(ws, _session.params)
                )

            elif action == "stop":
                await _session.cancel()
                await _send(ws, {"type": "stopped"})

            elif action == "update_params":
                _session.params.update(**msg.get("params", {}))

            else:
                await _send(ws, {"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        await _session.cancel()
    except Exception:
        logger.exception("WebSocket error")
        await _session.cancel()
