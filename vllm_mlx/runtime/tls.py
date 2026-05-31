# SPDX-License-Identifier: Apache-2.0
"""Ephemeral in-memory TLS for ``rapid-mlx serve`` (macOS).

When ``--self-signed-days > 0`` we mint a self-signed cert in process memory
and serve HTTPS with it, with zero disk access. stdlib ``ssl`` only loads
cert chains from a path, so the in-RAM PEM is fed to OpenSSL through an
anonymous ``os.pipe()`` referenced as ``/dev/fd/N``: the bytes live in a
kernel buffer and are never written to disk. On macOS ``open("/dev/fd/N")``
is equivalent to ``dup(N)`` (see ``fd(4)``), and our read-only open of the
pipe's read end satisfies the subset-of-mode rule, so OpenSSL reads straight
from the pipe. uvicorn can't take an ``SSLContext`` directly, so ``run()``
injects it as ``Config.ssl``. The context pins TLS 1.3 and disables session
tickets (no 0-RTT early data, no resumption).
"""

from __future__ import annotations

import logging
import os
import ssl
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _self_signed_pem(hosts: list[str], days: int) -> tuple[bytes, bytes]:
    """Return ``(cert_pem, key_pem)`` for a self-signed cert covering ``hosts``."""
    import ipaddress
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "rapid-mlx")])

    sans: list[x509.GeneralName] = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .sign(key, hashes.SHA256())
    )
    return (
        cert.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )


def _feed_and_close(fd: int, data: bytes) -> None:
    """Write ``data`` to a write-end fd, then close it to signal EOF."""
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _context_from_pem(cert_pem: bytes, key_pem: bytes) -> ssl.SSLContext:
    """Load in-memory PEM into a server ``SSLContext`` without touching disk.

    Each PEM is pushed into an anonymous ``os.pipe()`` (kernel buffer) and
    handed to OpenSSL as ``/dev/fd/N``; nothing is written to the filesystem.

    Pins TLS 1.3 and disables session tickets — without a ticket there is no
    PSK for a client to carry 0-RTT early data, so this forbids the
    (replayable) 0-RTT path along with session resumption.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.options |= ssl.OP_NO_TICKET
    read_fds: list[int] = []
    try:
        paths: list[str] = []
        for data in (cert_pem, key_pem):
            r, w = os.pipe()
            read_fds.append(r)
            threading.Thread(
                target=_feed_and_close, args=(w, data), daemon=True
            ).start()
            paths.append(f"/dev/fd/{r}")
        ctx.load_cert_chain(paths[0], paths[1])
    finally:
        for r in read_fds:
            try:
                os.close(r)
            except OSError:
                pass
    return ctx


def build_ephemeral_context(host: str | None, days: int) -> ssl.SSLContext:
    """Mint a fresh self-signed cert for the bind IP and return a context.

    SANs: the bind IP, always ``rapid-mlx.local``, and ``localhost`` when
    the bind IP is loopback (so ``https://localhost`` verifies too).
    """
    import ipaddress

    sans = [host] if host else ["localhost"]
    try:
        if host and ipaddress.ip_address(host).is_loopback:
            sans.append("localhost")
    except ValueError:
        pass
    sans.append("rapid-mlx.local")
    ctx = _context_from_pem(*_self_signed_pem(sans, days))
    logger.warning(
        f"TLS: ephemeral in-memory self-signed cert (valid {days}d, SANs={sans}, "
        "nothing on disk). Clients must trust it or skip verification."
    )
    return ctx


def require_bindable_ip(host: str, cert_days: int) -> None:
    """When TLS is on, require ``host`` to be a specific (non-wildcard) IP.

    The cert pins the bind IP via SAN, so a wildcard (0.0.0.0 / ::) or a
    hostname can't be covered. Raises ``SystemExit`` to fail early, before
    the model loads.
    """
    if cert_days <= 0:
        return
    import ipaddress

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise SystemExit(
            f"--self-signed-days requires --host to be a specific IP, got {host!r}"
        ) from None
    if ip.is_unspecified:
        raise SystemExit(
            f"--self-signed-days cannot bind the wildcard {host!r}; "
            "use a specific IP such as 127.0.0.1"
        )


def resolve_tls_context(host: str | None, days: int) -> ssl.SSLContext | None:
    """Return a server context when ``days > 0``, else ``None`` (plain HTTP)."""
    return build_ephemeral_context(host, days) if days > 0 else None


def run(app: Any, *, host: str, port: int, cert_days: int, **uvicorn_kwargs: Any) -> None:
    """Run uvicorn, enabling in-memory TLS when ``cert_days > 0``.

    Injects the context as ``Config.ssl`` (uvicorn only accepts cert paths,
    so we load the config then attach the context before serving).
    """
    import uvicorn

    from ..config import get_config

    ctx = resolve_tls_context(host, cert_days)
    # Stashed for the lifespan "Ready:" banner; attribute, not a declared
    # field, to avoid touching ServerConfig.
    get_config().bind_scheme = "https" if ctx else "http"

    config = uvicorn.Config(app, host=host, port=port, **uvicorn_kwargs)
    config.load()
    if ctx is not None:
        config.ssl = ctx
    uvicorn.Server(config).run()
