# Hailo deploy: converting an ONNX model to a Frigate-ready HEF

The final artifact Frigate's `hailo` detector consumes is a **HEF** (Hailo
Executable Format) file. This document is the real-world manual for the P12
`deploy` stage: turning a trained `best.onnx` (e.g.
`data/models/yolov9s/best.onnx`) into a HEF you can drop onto the Frigate NVR.

## Do you need a Hailo device?

**No.** The conversion is done entirely by the **Hailo Dataflow Compiler**
(DFC), host-side software: it parses the ONNX into a HAR, quantizes it (with
your calibration images), and compiles the quantized HAR into the HEF. A Hailo
device (with HailoRT) is only needed to *run* the HEF, or for on-hardware
accuracy validation — not to produce it.

## Where the conversion can run

The DFC officially supports **Ubuntu 22.04/24.04 x86-64 (or WSL2)** — there is
no macOS or ARM build. Options:

| Host                    | How                              |
| ----------------------- | -------------------------------- |
| x86-64 Linux (a VM, the NVR if x86-64) | install the DFC, `hailo` CLI on PATH |
| macOS (this repo's common case) | build `frigate-learn-hailo-dfc` (amd64) and run inside Docker via `hailo.docker_image` |
| ARM NVR / RPi           | compile elsewhere, copy the `.hef` over |

## Toolchain versions (Hailo-8)

Use the Hailo-8 line: **Dataflow Compiler 3.3x** (3.34.0 verified against this
repo) + Hailo Model Zoo **v2.19.0**. The current `master` branch / DFC 5.x
targets Hailo-10/15 only, and newer-DFC HEFs may not load under the HailoRT
version Frigate pins (4.21.x in current Frigate images). DFC supported archs in
3.34 are `hailo8`, `hailo8l`, `hailo8r` — the Frigate box is `hailo8`.

## One-time setup

1. Download the DFC wheel from the Hailo developer zone (Hailo Software Suite
   download), e.g. `hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl`,
   and place it in the repo root.

2. Build the container (macOS host):

   ```bash
   scripts/build-hailo-dfc.sh          # -> frigate-learn-hailo-dfc:3.34.0
   ```

   First build pulls TensorFlow/JAX and takes a while / several GB. Only needed
   once.

3. Configure `config.yaml`:

   ```yaml
   hailo:
     hw_arch: "hailo8"
     docker_image: "frigate-learn-hailo-dfc:3.34.0"
     calib_images: "data/images"       # real camera frames for quantization
     calib_samples: 64
     interactive: false                # auto-accept DFC prompts (-y)
   ```

   On an x86-64 Linux host, leave `docker_image: null` and install the DFC
   there instead.

## The compile (what `deploy --real` does)

Three DFC stages, run in the model dir (`calib.npy` is built from
`calib_images` first — letterboxed to `imgsz`, float32 NHWC 0..1):

```bash
hailo parser onnx best.onnx --net-name best --hw-arch hailo8 -y
hailo optimize best.har --hw-arch hailo8 --calib-set-path calib.npy \
    --output-har-path best_optimized.har
hailo compiler best_optimized.har --hw-arch hailo8 --output-dir .
```

Run it:

```bash
.venv/bin/frigate-learn deploy yolov9s data/models/yolov9s/best.onnx --real
# optional: --calib <dir> --hw-arch <arch> --docker-image <tag> --calib-samples N
```

Outputs in `data/models/yolov9s/`: `best.hef` (renamed to the model dir as
`yolov9s.hef` by default naming), `manifest.json`, `frigate-detector.yml`.

### NMS — the Frigate compatibility catch

Frigate's Hailo detector (`frigate/detectors/plugins/hailo.py`) expects **NMS
post-processed, per-class detections** (`[x1, y1, x2, y2, score]` per class),
not raw tensor outputs. The `-y` flag makes the DFC parser auto-detect the
ultralytics-style detection head on your export, re-parse at the raw head
convs, and append `nms_postprocess(...)` to the model script — so the HEF
comes out with **on-device NMS**, which is exactly what Frigate consumes. A
plain raw-tensor HEF (e.g. compiled with a generic tool without this step)
will load but produce garbage detection arrays in Frigate.

If your export isn't auto-detected (non-ultralytics layout, or the parser
prints node recommendations), follow the parser's suggested
`--start-node-names/--end-node-names`, then re-run; failing that, the Hailo
Model Zoo **v2.19.0** route (`hailomz parse/optimize/compile` with a network
YAML modeled on `hailo_model_zoo/cfg/networks/yolov9c.yaml` — note that config
expects the 6-output WongKinYiu v9 layout, not the single-output ultralytics
export) is the official fallback with full evaluation tooling.

## Calibration matters

Quantization needs representative input. Point `hailo.calib_images` at a
folder of real camera frames (the collector's `data/images` works; a few
hundred frames is plenty). Without it, `deploy --real` still compiles using
random calibration data (`--use-random-calib-set`) but prints a loud warning —
the HEF's quantized accuracy is untrustworthy.

## Verify before shipping

- With a device attached (the NVR): `hailortcli run <hef>` or
  `hailo runtime-profiler` should load and infer the 320×320 input.
- On the host: `hailo profiler best_optimized.har` prints expected
  performance/accuracy estimates.
- Sanity-check the manifest: `hef_pending` should be `false`, `hw_arch`
  `hailo8`, `hef_sha256` populated. Then copy the HEF to
  `/usr/share/frigate/models/` on the NVR and paste `frigate-detector.yml`
  under `detectors:` in Frigate's config.

## Troubleshooting

- **`HailoCompilerMissing` on macOS:** `hailo.docker_image` unset — build and
  set it; or Docker Desktop missing.
- **Parser rejects ops / prints recommendations:** YOLOv9's reparameterizable
  `RepConv` blocks are the usual culprit — re-export from a fused/deployed
  model (`model.fuse()` before export) so the ONNX contains only folded
  convs, then follow the parser's suggested node names.
- **Compile is slow on Apple Silicon:** the amd64 container runs under
  emulation; quantization is CPU-only inside it. Acceptable for a one-shot;
  for regular runs, use an x86-64 Linux box.
- **HEF won't load in Frigate (HailoRT error):** HEF/RT version mismatch —
  stick to DFC 3.3x (this repo's default) or the exact DFC used to produce
  the compiled-zoo HEFs Frigate downloads.