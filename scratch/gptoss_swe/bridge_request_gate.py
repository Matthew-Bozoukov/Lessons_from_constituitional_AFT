# ABOUTME: Temporary byte-preserving TCP queue for a drained localhost bridge handoff.
# ABOUTME: No HTTP parsing, model calls, retries or credential persistence; routing excludes the gate UID.
import asyncio
import json
import os
import time
from pathlib import Path


class Gate:
    def __init__(self, control, status):
        self.control, self.status = Path(control), Path(status)
        self.accepted = self.connected = self.finished = 0
        self.active = self.queued = 0
        self.errors = []

    async def upstream(self):
        while True:
            value = json.loads(self.control.read_text())
            port = value.get("port")
            # Before the bridge is touched, an expired drain lease returns new
            # requests to the original. Once replacement begins no fail-open.
            if (
                port is None
                and value.get("fallback_port")
                and time.time() > value["lease_until"]
            ):
                port = value["fallback_port"]
            if port is not None:
                return int(port)
            await asyncio.sleep(0.05)

    async def handle(self, reader, writer):
        self.accepted += 1
        self.active += 1
        self.queued += 1
        upstream_writer = None
        try:
            port = await self.upstream()
            upstream_reader, upstream_writer = await asyncio.open_connection(
                "127.0.0.1", port
            )
            self.connected += 1
            self.queued -= 1

            async def pipe(src, dst):
                while data := await src.read(65536):
                    dst.write(data)
                    await dst.drain()
                if dst.can_write_eof():
                    dst.write_eof()

            await asyncio.gather(
                pipe(reader, upstream_writer), pipe(upstream_reader, writer)
            )
            self.finished += 1
        except Exception as exc:
            self.errors.append(type(exc).__name__)
        finally:
            self.active -= 1
            writer.close()
            if upstream_writer:
                upstream_writer.close()

    async def report(self):
        while True:
            value = dict(
                at=time.time(),
                pid=os.getpid(),
                accepted=self.accepted,
                connected=self.connected,
                finished=self.finished,
                active=self.active,
                queued=self.queued,
                errors=self.errors,
            )
            tmp = self.status.with_suffix(".tmp")
            tmp.write_text(json.dumps(value))
            os.replace(tmp, self.status)
            await asyncio.sleep(0.1)

    async def run(self, port):
        server = await asyncio.start_server(self.handle, "127.0.0.1", port, backlog=256)
        async with server:
            await asyncio.gather(server.serve_forever(), self.report())


if __name__ == "__main__":
    import sys

    asyncio.run(Gate(sys.argv[2], sys.argv[3]).run(int(sys.argv[1])))
