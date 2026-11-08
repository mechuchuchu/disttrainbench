"""Apply a temporary tc setup to one shared Docker network namespace."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import secrets
import subprocess
import sys


INTERFACE = "eth0"


def interface_subnets(interface: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    result = subprocess.run(
        ["ip", "-o", "addr", "show", "dev", interface],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"could not inspect {interface}")
    subnets = []
    for line in result.stdout.splitlines():
        match = re.search(r"\sinet6?\s+(\S+)", line)
        if not match:
            continue
        address = ipaddress.ip_interface(match.group(1))
        if not (
            address.ip.is_link_local
            or address.ip.is_loopback
            or address.ip.is_unspecified
        ):
            if address.network not in subnets:
                subnets.append(address.network)
    if not subnets:
        raise RuntimeError(f"no Docker bridge subnet found on {interface}")
    return subnets


def run_tc(arguments: list[str]) -> None:
    result = subprocess.run(
        ["tc", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "tc failed"
        raise RuntimeError(detail)


def rate_string(rate_bps: int) -> str:
    if rate_bps % 1000 == 0:
        return f"{rate_bps // 1000}kbit"
    return f"{rate_bps}bit"


def shape(
    interface: str,
    subnets: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    rate_bps: int,
    latency_ms: int,
) -> None:
    rate = rate_string(rate_bps)
    burst_bytes = max(4096, rate_bps // 10)

    run_tc([
        "qdisc", "replace", "dev", interface, "root", "handle", "1:", "prio",
        "bands", "2", "priomap", *("1" for _ in range(16)),
    ])
    run_tc([
        "qdisc", "replace", "dev", interface, "parent", "1:1", "handle", "10:",
        "tbf", "rate", rate, "burst", str(burst_bytes), "limit", "128000",
    ])
    run_tc([
        "qdisc", "replace", "dev", interface, "parent", "10:1", "handle", "20:",
        "netem", "limit", "128", "delay", f"{latency_ms}ms",
    ])
    for index, subnet in enumerate(subnets, start=10):
        protocol = "ip" if subnet.version == 4 else "ipv6"
        ip_match = "ip" if subnet.version == 4 else "ip6"
        run_tc([
            "filter", "replace", "dev", interface, "protocol", protocol, "parent", "1:",
            "prio", str(index), "u32", "match", ip_match, "dst", str(subnet),
            "flowid", "1:1",
        ])
    run_tc(["qdisc", "replace", "dev", interface, "handle", "ffff:", "ingress"])
    for index, subnet in enumerate(subnets, start=10):
        protocol = "ip" if subnet.version == 4 else "ipv6"
        ip_match = "ip" if subnet.version == 4 else "ip6"
        run_tc([
            "filter", "replace", "dev", interface, "parent", "ffff:", "protocol", protocol,
            "prio", str(index), "u32", "match", ip_match, "src", str(subnet),
            "police", "rate", rate, "burst", str(burst_bytes), "drop",
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-bps", type=int, required=True)
    parser.add_argument("--latency-min-ms", type=int, default=1)
    parser.add_argument("--latency-max-ms", type=int, default=700)
    parser.add_argument("--interface", default=INTERFACE)
    args = parser.parse_args()
    if args.rate_bps <= 0:
        parser.error("--rate-bps must be positive")
    if args.latency_min_ms < 1 or args.latency_max_ms < args.latency_min_ms:
        parser.error("latency range must satisfy 1 <= minimum <= maximum")
    latency_ms = args.latency_min_ms + secrets.randbelow(
        args.latency_max_ms - args.latency_min_ms + 1
    )
    try:
        subnets = interface_subnets(args.interface)
        shape(args.interface, subnets, args.rate_bps, latency_ms)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"network shaping failed: {error}", file=sys.stderr, flush=True)
        return 1
    print("DISTTRAINBENCH_NETWORK_SHAPER=" + json.dumps({
        "event": "network_shaping_configured",
        "interface": args.interface,
        "subnets": [str(subnet) for subnet in subnets],
        "rate_bits_per_second": args.rate_bps,
        "egress_latency_ms": latency_ms,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
