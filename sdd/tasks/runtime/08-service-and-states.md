08 - Service and Residency States

Status: done
Commits:

## Goal

Make `load`/`unload` real. `zlm load <alias>` spawns a small background
**service** that holds the runtime's Session (image mmap, buffers, KV) and serves
generation over a local socket. `zlm run <alias>` uses that service and, when the
alias is not online, **loads it internally first** (printing a notice in the
terminal) — so `load` is optional (preload). The service has a **residency
state** while up — `cold` (weights on disk), `warm` (page cache), `hot` (pinned)
— with an idle TTL that demotes. `zlm image list` reports online/offline;
`unload` stops the service. A first, minimal step toward the session/daemon of
the earlier local prototype (which ran a socket.io sub-daemon).

## Scope

- **In:** a per-**plan** service process (keyed by `<image>_<fp>`; N aliases
  share it); a Unix-domain socket + pid/log state under `.zlm/runtimes/`; `load`
  spawns it (detached) and waits until online; `run`
  connects and delegates generation, and **auto-loads** (spawns the service,
  printing a notice) when offline; a residency state (`cold`/`warm`/`hot`, default
  **warm**) set initially by `load --cold` / `--warm` / `--hot[=minutes]`, plus
  `--timeout=minutes` (service auto-unload after idle); `image list` SERVICE =
  online/offline; `unload` stops the service; liveness = socket connect/ping.
- **Out:** multi-client sessions, auth, TCP/network, GPU; concurrent sessions;
  the full `session` (conversation) model; the rich `zlm state` command (future —
  `load` sets the initial state).

## Design

- **State.** `.zlm/runtimes/<image>_<fp>.socket` (mode 0600) plus
  `<image>_<fp>.pid`/`.log`; the dir is 0700. Online iff the socket exists and a
  `PING` round-trips. The socket sits with the `.zplan`/`.zat` (same key): one
  daemon per plan, shared by every alias that resolves to it.
- **Daemon.** `load` re-execs zlm as a detached process (fork + `setsid`,
  stdout/stderr to the log) running a hidden `zlm __serve <alias>`: it resolves
  the alias to its plan, opens the `Session`, binds the socket and loops. It
  owns the preloaded model — the whole point of `load`.
- **Protocol (line-framed, minimal).** Requests: `PING`, `GENERATE\t<prompt>\t<max_tokens>`,
  `QUIT`. Replies: `PONG`, `OK\t<byte_len>\n<bytes>`, `ERR\t<msg>`.
- **`run`.** Resolves the alias to its plan socket; if not online, it **loads it
  internally** (spawns the service, printing e.g. `loading '<alias>'...`) and
  then connects. Sends `GENERATE` and prints the reply — the edge still speaks
  text only. (So `run` standalone works; `load` only preloads.)
- **`image list` SERVICE.** A row is `online` if some alive service references a
  plan whose `image` equals that image (read from the plan / pid record);
  otherwise `offline`. (Until `engine/03`, the service is per runtime alias under
  `runtimes/<alias>/`.)
- **State & lifetime.** While the service is up it holds a residency state:
  - **cold** — service up; weights on disk only (window 0, no prefetch retention
    to our accounting); buffer/KV `fixed` still resident.
  - **warm** — service up; weights warmed into the **page cache** (readahead
    granted) but not pinned/accounted.
  - **hot** — service up; the plan's value-ranked set **pinned** within the
    window.
  `load` **defaults to `warm`**; `--cold` / `--warm` / `--hot[=minutes]` set the
  initial state, where `--hot=minutes` is the **pin TTL** (after N idle minutes,
  demote `hot -> warm`; no minutes = hold). `--timeout=minutes` is the **service
  lifetime TTL** (auto-unload after N idle minutes; orthogonal to the state); the
  default is **30 min** (`default_timeout_minutes`), and `--timeout=0` opts out
  (hold until an explicit `unload`). `run`'s auto-load uses the default (warm,
  30 min). **Mapping to the residency engine:** cold = window 0, no readahead;
  warm = window 0 + prefetch readahead (page cache); hot = window > 0 + pin the
  plan's ranked set. A future `zlm state <alias> [--cold|--warm|--hot=minutes]`
  changes/reports it live (`state` with no flag = status). The runtime still
  decides nothing: `state` sets the window; the plan's value ranking picks the
  ranges.
- **`unload`.** Stops the service (lifecycle, not a residency state): sends
  `QUIT` (fallback SIGTERM by pid), waits, removes the socket/pid — `image list`
  goes `offline`. With a **shared** service, `unload <alias> --remove` stops it
  only when it is the **last alias** for that plan (refcount); otherwise the
  daemon stays. Ties into `engine/03`'s store `--remove`/`--prune` flags.
- **Runtime unchanged in spirit:** the service only holds the `Session` and
  applies the same `generate` the edge would; it decides nothing.

## Flow / Arch

```text
load <alias>  -> spawn detached `zlm __serve <alias>` -> wait PING -> "loaded"
__serve       -> Session.open(plan); bind sock; loop { PING | GENERATE | QUIT }
run <alias>   -> offline ? spawn service (print "loading...") ; connect -> GENERATE -> print
image list    -> per image: any alive service with that image ? online : offline
unload        -> sock QUIT / SIGTERM ; cleanup
```

## Arch Provenance

- **own:** the per-plan service + socket protocol; the AoT runtime stays the
  same (the service reuses `Session.generate`).
- **spec:** POSIX Unix-domain sockets.
- **consulted:** the earlier local prototype's `load` sub-daemon (socket.io) — reference only.
- **derived:** from `runtime/` (Session) and `image list`'s reserved SERVICE
  column; `zrun` reserves fields for this.

## Steps

1. State files: `.zlm/runtimes/<image>_<fp>.socket|.pid|.log` paths + perms;
   liveness (`PING`).
2. Hidden `zlm __serve <alias>`: open the Session, bind the socket, serve
   `PING`/`GENERATE`/`QUIT`; clean socket/pid on exit.
3. Residency state (`cold`/`warm`/`hot`) + TTLs in the service: set the window
   (cold 0, warm readahead, hot pin the plan's ranked set); `--hot=minutes`
   demotes `hot -> warm` on idle; `--timeout=minutes` auto-unloads the service.
4. `load`: spawn detached in the default state (**warm**);
   `--cold`/`--warm`/`--hot[=minutes]` + `--timeout=minutes`; wait for online;
   idempotent when already online.
5. `run`: resolve alias -> socket; when offline, spawn the service (printing a
   notice) and wait, then delegate `GENERATE`. `load` only preloads.
6. `image list`: SERVICE online/offline from alive services.
7. `unload`: stop the service (QUIT/SIGTERM) + cleanup.
8. Tests: protocol round-trip; liveness; state/TTL demotion; socket cleanup on
   `QUIT`; `run` auto-loads when offline.
9. `zig build` / `zig build test` green; e2e: `run` with no prior `load`
   auto-loads (printing) -> `image list` online -> `load --hot=1` pins -> idle
   demotes -> `unload` -> offline.

## Definition of Done

- [x] `load <alias>` spawns a service that holds the Session and serves generation.
- [x] `run <alias>` works standalone: auto-loads (printing) when offline; `load`
      only preloads.
- [x] The service holds a residency state (cold/warm/hot); `--hot=minutes` demotes
      to warm on idle.
- [x] `load` defaults to `warm`; `--cold`/`--warm`/`--hot[=minutes]` set the
      state; `--timeout=minutes` auto-unloads the service.
- [x] `image list` reports online/offline per image.
- [x] `unload` stops the service and cleans the socket/pid.
- [x] `zig build` / `zig build test` green; e2e load/run/unload works.

**As-built (states + TTLs).**
- States map to the residency engine: **cold** window 0 + no prefetch; **warm**
  (default) window 0 + prefetch (readahead); **hot** `plan.window_bytes` +
  prefetch (pin). `Residency.setState`; the service applies it at start.
- TTLs via `poll(250ms)` + monotonic idle clock: `--hot=MIN` demotes hot->warm;
  `--timeout=MIN` auto-unloads (`QUIT`, cleans socket/pid), defaulting to
  `default_timeout_minutes` (30) when unset and `--timeout=0` holding forever.
  `load`/`run`/`__serve` parse and pass `--state`/`--hot`/`--timeout`;
  already-online `load` is a no-op.
- Verified: `--cold` run coherent; `--timeout=1` auto-unloads (offline, artifacts
  removed); demote keeps online; 135/135 tests.
- Liveness: the daemon self-exits when its socket inode is gone/replaced or the
  pid file no longer names it (idle poll check); `unload --prune` SIGTERMs
  orphaned services (plan dir with no alias). It also self-exits when the
  **registry no longer references its plan** (checked every ~5 s on the idle
  poll), so a re-tune that supersedes the plan releases the model without
  `--prune`. Idle auto-unload defaults to 30 min (`--timeout=0` holds).

## Comments

- Design decisions (confirm before implementing): **Unix-domain socket** (POSIX,
  not TCP), **hidden `__serve` subcommand** (re-exec, detached), **`run`
  delegates generation to the service**, and **`run` auto-loads when offline**
  (printing a notice; `load` only preloads). Adjust if you prefer otherwise.
- `state` (residency) vs `unload` (lifecycle) are distinct: `state` always has a
  live service; `unload` stops it. The live `zlm state` command is future work
  (`open-items`).
- **Session groundwork:** the service owns a persistent `Session`, which is the
  prerequisite for a real conversation/session (KV reuse, message history,
  id/resume). Phase 1 is stateless per request (reset `pos`); the full session is
  a follow-up (`runtime/10`/milestone `session`) that extends the same daemon.
- Depends on alias resolution; until `engine/03`, uses `runtimes/<alias>/`.
- Security: socket 0600 + dir 0700; single user, no auth.
- `zrun`'s reserved `flags` are the intended home for the socket/session/refcount
  once `engine/03` lands.
