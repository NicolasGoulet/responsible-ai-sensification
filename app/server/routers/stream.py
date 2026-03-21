"""stream.py: WebSocket endpoint for live pipeline streaming."""
import asyncio
import dataclasses
import json
import logging

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


async def _run_pipeline(ws: WebSocket, params: PipelineParams) -> None:
    """Background task: load model (cached), run inspect_live, stream token events."""
    try:
        cache_key = (params.model, params.layer, params.width)

        if cache_key in _model_cache:
            await _send(ws, {"type": "loading", "stage": "cached"})
            cached = _model_cache[cache_key]
            model = cached["model"]
            tokenizer = cached["tokenizer"]
            sae = cached["sae"]
            neuronpedia = cached["neuronpedia"]
        else:
            # Load model
            await _send(ws, {"type": "loading", "stage": "Loading language model..."})

            def _load_model():
                from transformers import AutoModelForCausalLM, AutoTokenizer
                _model = AutoModelForCausalLM.from_pretrained(params.model, device_map="auto")
                _tokenizer = AutoTokenizer.from_pretrained(params.model)
                return _model, _tokenizer

            model, tokenizer = await asyncio.to_thread(_load_model)

            # Load SAE
            await _send(ws, {"type": "loading", "stage": f"Loading SAE (layer={params.layer}, width={params.width})..."})

            def _load_sae():
                from app.server.pipeline.extract import load_sae
                return load_sae(layer=params.layer, width=params.width, l0=params.l0)

            sae = await asyncio.to_thread(_load_sae)

            # Load Neuronpedia
            await _send(ws, {"type": "loading", "stage": "Downloading Neuronpedia explanations..."})

            def _load_neuronpedia():
                from app.server.pipeline.extract import download_neuronpedia_explanations
                # Neuronpedia uses "gemma-3-1b" style key, strip the HF org prefix
                np_model_id = params.model.split("/")[-1].replace("-pt", "")
                return download_neuronpedia_explanations(np_model_id, params.layer, params.width)

            neuronpedia = await asyncio.to_thread(_load_neuronpedia)

            _model_cache[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "sae": sae,
                "neuronpedia": neuronpedia,
            }

        await _send(ws, {"type": "loading", "stage": "Running generation..."})

        from app.server.pipeline.extract import inspect_live
        from app.server.pipeline.transform import apply_identity, apply_cluster, build_cluster_map

        cluster_map: dict = {}
        if params.strategy == "cluster":
            cluster_key = (params.model, params.layer, params.width, params.clusters)
            if cluster_key in _cluster_cache:
                cluster_map = _cluster_cache[cluster_key]
            else:
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

        # Producer + synthesizer: run generation concurrently with timed playback
        queue: asyncio.Queue = asyncio.Queue()
        collected: list[dict] = []
        event_loop = asyncio.get_event_loop()

        async def _producer():
            def _generate():
                for token_analysis, elapsed_ms in inspect_live(
                    params.prompt, model, tokenizer, sae,
                    params.layer, neuronpedia,
                    max_new_tokens=params.max_tokens,
                ):
                    active_features = [f.model_dump() for f in token_analysis.active_features]
                    if params.strategy == "cluster":
                        notes = apply_cluster(active_features, cluster_map)
                    else:
                        notes = apply_identity(active_features)
                    event = {
                        "type": "token",
                        "token": token_analysis.token,
                        "token_id": token_analysis.token_id,
                        "elapsed_ms": elapsed_ms,
                        "notes": notes,
                    }
                    asyncio.run_coroutine_threadsafe(queue.put(event), event_loop).result()

            await asyncio.to_thread(_generate)
            await queue.put(None)  # sentinel: generation complete

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
