# The container runs the same binary as the host; build it statically first:
#   zig build -Dtarget=x86_64-linux-musl -Doptimize=ReleaseFast -Dcpu=native
FROM alpine:3

COPY zig-out/bin/zlm /usr/local/bin/zlm

WORKDIR /work

ENTRYPOINT ["/bin/sh", "-c"]
