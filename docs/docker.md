# Running zlm in Docker

This is an example workflow, not a supported deployment target. Docker is useful
for isolating a benchmark, reproducing a constrained cgroup, or running the
static Linux binary on a host without installing it.

The provided [`docker/Dockerfile`](../docker/Dockerfile) is intentionally small:
it copies an already-built static `zlm` binary into Alpine. Build the binary on
the host first.

## 1. Build the static binary

```bash
# Portable baseline for x86_64 Linux (same target as the release workflow).
zig build -Dtarget=x86_64-linux-musl -Doptimize=ReleaseFast -Dcpu=x86_64_v2

# Or optimize for this exact machine (not portable to older CPUs).
zig build -Dtarget=x86_64-linux-musl -Doptimize=ReleaseFast -Dcpu=native
```

## 2. Build the container image

```bash
docker build -f docker/Dockerfile -t zlm:local .
```

Expected (BuildKit): the build finishes with a line naming
`docker.io/library/zlm:local`:

```text
=> => naming to docker.io/library/zlm:local
```

The build context is the repository root, and `.dockerignore` only lets
`zig-out/bin/zlm` through. The container does not compile Zig.

## 3. Prepare a store and a tuned alias

Use a host directory as `ZLM_HOME`, then compile and tune before starting the
container:

```bash
export ZLM_HOME="$PWD/docker/home"
mkdir -p "$ZLM_HOME"

./zig-out/bin/zlm image compile \
  gemma4-e2b models/gemma-4-E2B-it-Q4_0.gguf
./zig-out/bin/zlm tune gemma4-e2b e2b --threads 8 --ram 8gb
```

`docker/home/` is ignored by `docker/.gitignore`; it can grow to many GB
because it holds the compiled image and runtime plans.

## 4. Run one-shot generation

The Dockerfile uses `/bin/sh -c` as its entrypoint. Override it with the binary
so arguments are passed directly:

```bash
docker run --rm -it \
  --network none \
  --init \
  --entrypoint /usr/local/bin/zlm \
  -e ZLM_HOME=/work/home \
  -v "$PWD/docker/home:/work/home" \
  -v "$PWD/models:/models:ro" \
  zlm:local run e2b \
    --soul "You are a concise assistant." \
    --prompt "Hello" \
    --max-tokens 64 \
    --metrics
```

`--network none` matches the benchmark setup. It is optional for local use, but
it guarantees the runtime does not reach the network.

Expected: the generated answer on stdout, followed by the `metrics:` block
because `--metrics` was passed:

```text
<generated text>

metrics:
  ttft:       <duration>
  generation: <duration>  <tok/s>  (<tokens> tok)
  prefill:    <duration>  <tok/s>  (<tokens> tok)
  decode:     <duration>  <tok/s>  (<tokens> tok)
  peak rss:   <kB> kB
```

## 5. Constrain memory

Docker's `--memory` and `--memory-swap` map to the cgroup limit used by the
survival benchmarks in [`benchmark-cpu.md`](benchmark-cpu.md):

```bash
# Tune and run inside a 1 GB container.
docker run --rm -it \
  --network none \
  --memory=1g --memory-swap=1g \
  --entrypoint /usr/local/bin/zlm \
  -e ZLM_HOME=/work/home \
  -v "$PWD/docker/home:/work/home" \
  -v "$PWD/models:/models:ro" \
  zlm:local tune e2b t1g --threads 8 --ram 1gb

docker run --rm -it \
  --network none \
  --memory=1g --memory-swap=1g \
  --entrypoint /usr/local/bin/zlm \
  -e ZLM_HOME=/work/home \
  -v "$PWD/docker/home:/work/home" \
  -v "$PWD/models:/models:ro" \
  zlm:local run t1g --prompt "Hello" --max-tokens 8 --metrics
```

`--memory-swap` equal to `--memory` disables swap and makes the limit strict.
For the host-tuned setup (plan generated without the cap, execution inside the
cap), run only the second command.

## 6. HTTP API inside the container

`zlm http serve` binds the host in its JSON config. Create a config, set
`"host": "0.0.0.0"` for the container, and publish the port:

```bash
./zig-out/bin/zlm http config --output docker/home/http.json
# Edit docker/home/http.json and change "host" to "0.0.0.0".

docker run --rm \
  -p 127.0.0.1:8080:8080 \
  --entrypoint /usr/local/bin/zlm \
  -e ZLM_HOME=/work/home \
  -v "$PWD/docker/home:/work/home" \
  zlm:local http serve 8080 --config=/work/home/http.json
```

Do not expose the API on a public interface unless you have reviewed the
limitations in [`sdd/open-items.md`](../sdd/open-items.md).

## 7. Optional: llama.cpp baseline

For a side-by-side benchmark, run the same GGUF and thread count in the
llama.cpp container (also with no network):

```bash
docker run --rm --network none \
  -v "$PWD/models:/models:ro" \
  --entrypoint /app/llama-completion \
  ghcr.io/ggml-org/llama.cpp:full \
    -m /models/gemma-4-E2B-it-Q4_0.gguf \
    -p "Hello" -n 64 --temp 0 -t 8 -c 1024 -no-cnv --ignore-eos
```

Run one container at a time so the two runtimes do not contend for CPU.

## Notes

- This workflow always runs the **Linux** binary, including on macOS Docker
  Desktop. A native macOS build is experimental; see the platform table in the
  [README](../README.md#build).
- The store must be writable: `tune`, the prefix cache and sessions write under
  `ZLM_HOME`.
- To remove the store, stop the containers and delete `docker/home/`.
