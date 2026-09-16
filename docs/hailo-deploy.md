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
    --model-script nms.alls --output-har-path best_optimized.har
hailo compiler best_optimized.har --hw-arch hailo8 --output-dir .
```

The `-y` flag at `parser` auto-detects the ultralytics-style detection head
and appends `nms_postprocess(...)` to the stored model script, which `optimize`
then applies. When auto-detection misses the head (fine-tuned renames, or the
yolov9 RepConv-y variant), inject the NMS explicitly with a `--model-script` at
the `optimize` stage as shown.

`nms.alls` is a one-line model script that injects the detection-head NMS the
Frigate detector needs:

```text
nms_postprocess("nms_config.json", meta_arch=yolov8)
```

with `nms_config.json` describing the head (image 320x320, 80 classes, the
yolov9 family uses `regression_length: 16`, and DFL-style scalar-per-stride
bbox decoders). See `data/models/yolov9s/nms-compile/nms.alls` +
`nms_config.json` for a known-good pair.

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
not raw tensor outputs. The DFC must embed that NMS **inside the HEF as
net-flow metadata** (`HAILO_NET_FLOW`, `HAILO_NET_FLOW_YOLOV8_NMS`,
`postprocess_output_layer` strings in the file, produced when `nms_metadata`
is set with `engine=cpu`); HailoRT 4.21 (what Frigate pins) then exposes a
single `yolov8_nms_postprocess` output stream (e.g. shape `[40080]` =
80 classes x 5 fields x 100 proposals) that the plugin reads. A raw-tensor HEF
loads fine but produces garbage detection arrays.

**The critical ordering: inject `nms_postprocess` at the `optimize` stage via
`--model-script`, *not* at the `compiler` stage.** Adding the same script only
to `hailo compiler` makes DFC 3.34 emit the post-process as a **separate
external ONNX** (`<name>_postprocess.onnx`) inside the HAR — the HailoRT 5.x
"on-the-fly post-process" mechanism — which HailoRT 4.21 ignores. The HEF then
comes out with the raw head convs (6 output streams for a yolo-yolov9 head,
`(40,40,64/80)`, `(20,20,64/80)`, `(10,10,64/80)` at 320 input) and no embedded
NMS, so Frigate/`hailo8l.py` never sees scores. When NMS is applied at
`optimize` instead, `nms_metadata` (engine `cpu`) survives into the quantized
HAR and the final HEF matches the stock compiled-zoo output.

Sanity-check a suspect HEF with HailoRT on the NVR before shipping:
`create_infer_model(hef).outputs` must list **one** named
`*_nms_postprocess` output (like stock's `[40080]`), not six conv tensors;
`hailortcli parse-hef` should show an `Op YOLOV8` / `YOLOV8-Post-Process`
operation. Grepping the raw bytes for `HAILO_NET_FLOW_YOLOV8_NMS` is a quick
host-side proxy. Note `hailo parser` has **no** `--model-script` option — NMS
can only be injected at optimize or compile.

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
- On the Frigate box (0.18 docker, this repo's case): confirm the HEF exposes
  a single NMS output stream with HailoRT 4.21 before restarting anything:

  ```python
  from hailo_platform import VDevice
  with VDevice() as v:
      im = v.create_infer_model("/tmp/<name>.hef")
      print([(o.name, o.shape) for o in im.outputs])  # want [('*', [40080])]
  ```

  then copy it into the config volume as `/config/model_cache/<name>.hef`
  (frigate 0.18 loads detectors from `model_cache/`, not `/usr/share/frigate`),
  and wire it under `model:` in Frigate's config:
  `width/height` = `imgsz`, `input_tensor: nhwc`, `input_pixel_format: rgb`,
  `input_dtype: int`, `model_type: yolo-generic`, `labelmap_path`, and
  `path: /config/model_cache/<name>.hef`. Keep `detectors:` (`type: hailo8l_siglip`,
  `device: PCIe`) untouched. Restart frigate and confirm `inference_speed` in
  `/api/stats` is single-digit/teen ms and `detection_fps` > 0 on cameras.
- Sanity-check the manifest: `hef_pending` should be `false`, `hw_arch`
  `hailo8`, `hef_sha256` populated.

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
- **HEF loads but detections are garbage / zero events:** a raw-tensor HEF with
  no embedded NMS. Embedding happens only when `nms_postprocess` is applied at
  the **optimize** stage (`--model-script nms.alls`, or `-y` auto-detection
  carried through to optimize). Injecting it only to `hailo compiler` silently
  produces a separate external postprocess ONNX that HailoRT 4.21 (Frigate's
  pinned runtime) ignores — the HEF exposes six conv output streams instead of
  one `*_nms_postprocess`. Confirm with `create_infer_model(hef).outputs`: you
  want one `[40080]`-shaped output, not six `(40,40,64/80)`-style tensors.
  DFC 3.34 emits net-flow metadata (`nms_metadata`, engine `cpu`) only when the
  NMS is in the HAR before compile — the compiled HAR's stored alls should read
  `nms_postprocess("{har}", meta_arch=yolov8)` and loading it via ClientRunner
  must yield a non-`None` `nms_metadata`. Both this repo's working HEF and the
  stock compiled-zoo `yolov8l_hailo8.hef` are proto version 5 / DFC 3.34 — the
  discriminator is the embedded net-flow metadata, not the HEF version.
- **HEF won't load in Frigate (HailoRT error):** HEF/RT version mismatch —
  stick to DFC 3.3x (this repo's default) or the exact DFC used to produce
  the compiled-zoo HEFs Frigate downloads.