# SPDX-License-Identifier: Apache-2.0
"""Pin the ephemeral in-memory TLS contract for ``rapid-mlx serve``.

Two invariants matter and are easy to regress:

1. **It actually serves TLS** — ``resolve_tls_context`` must return a
   usable server ``SSLContext`` that completes a handshake, with the
   minted cert covering ``localhost``/loopback via SubjectAltName.
2. **Zero disk access for the secret** — no cert/key bytes touch the
   filesystem. The loader feeds PEM to OpenSSL through an anonymous
   ``os.pipe()`` referenced as ``/dev/fd/N``, leaving no artifact behind.
   If someone "simplifies" the loader back to a temp file, the
   leftover-glob assertion fails.

The TLS path is pure stdlib + ``cryptography`` (no MLX/Metal needed).
"""

from __future__ import annotations

import glob
import os
import socket
import ssl
import tempfile
import threading

import pytest

from vllm_mlx.runtime import tls


def _serve_once(ctx: ssl.SSLContext, payload: bytes = b"OK") -> int:
    """Bind a loopback socket, serve one TLS connection in a thread.

    Returns the bound port. The background thread accepts a single
    connection, wraps it with ``ctx``, sends ``payload``, and closes.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(5)
    port = srv.getsockname()[1]

    def _serve() -> None:
        try:
            conn, _ = srv.accept()
            with ctx.wrap_socket(conn, server_side=True) as s:
                s.sendall(payload)
        except OSError:
            pass
        finally:
            srv.close()

    threading.Thread(target=_serve, daemon=True).start()
    return port


def _client_recv(port: int, cadata: str | None = None) -> bytes:
    """Connect over TLS and return the server's reply.

    When ``cadata`` is given the server cert is verified against it (and
    the hostname against the SAN); otherwise verification is disabled.
    """
    if cadata is not None:
        cctx = ssl.create_default_context(cadata=cadata)
    else:
        cctx = ssl.create_default_context()
        cctx.check_hostname = False
        cctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with cctx.wrap_socket(raw, server_hostname="localhost") as t:
            return t.recv(64)


def _cert_sans(ctx: ssl.SSLContext) -> list[str]:
    """Return the SAN values of the cert ``ctx`` actually serves."""
    from cryptography import x509

    port = _serve_once(ctx)
    cctx = ssl.create_default_context()
    cctx.check_hostname = False
    cctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with cctx.wrap_socket(raw, server_hostname="localhost") as t:
            der = t.getpeercert(binary_form=True)
    cert = x509.load_der_x509_certificate(der)
    ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return [str(g.value) for g in ext]


def _no_tls_tempdirs() -> list[str]:
    """All stray ``rapid-mlx-tls-*`` dirs across likely temp roots."""
    roots = {tempfile.gettempdir(), "/tmp"}
    found: list[str] = []
    for root in roots:
        found += glob.glob(os.path.join(root, "rapid-mlx-tls-*"))
    return found


def test_resolve_returns_none_when_days_zero():
    """``--self-signed-days 0`` → plain HTTP (None), serve path unchanged."""
    assert tls.resolve_tls_context("127.0.0.1", 0) is None


def test_resolve_returns_server_context_when_days_positive():
    ctx = tls.resolve_tls_context("127.0.0.1", 30)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.protocol == ssl.PROTOCOL_TLS_SERVER


def test_ephemeral_context_completes_handshake():
    """End-to-end: the minted context serves a real TLS connection."""
    ctx = tls.build_ephemeral_context(host="127.0.0.1", days=30)
    port = _serve_once(ctx)
    assert _client_recv(port) == b"OK"


def test_loopback_ip_adds_localhost_san():
    """Binding 127.0.0.1 must let a verifying client reach it as localhost."""
    sans = _cert_sans(tls.build_ephemeral_context(host="127.0.0.1", days=30))
    assert "127.0.0.1" in sans and "localhost" in sans


def test_non_loopback_ip_has_no_localhost_san():
    """A routable IP gets only itself (+ rapid-mlx.local) — no localhost."""
    sans = _cert_sans(tls.build_ephemeral_context(host="192.168.1.50", days=30))
    assert "192.168.1.50" in sans and "localhost" not in sans


@pytest.mark.parametrize("host", ["127.0.0.1", "192.168.1.50"])
def test_rapid_mlx_local_always_in_san(host):
    """rapid-mlx.local is always present regardless of bind IP."""
    sans = _cert_sans(tls.build_ephemeral_context(host=host, days=30))
    assert "rapid-mlx.local" in sans


@pytest.mark.parametrize("bad_host", ["0.0.0.0", "::", "example.com", "not-an-ip"])
def test_require_bindable_ip_rejects(bad_host):
    """TLS on + wildcard/hostname → exit early."""
    with pytest.raises(SystemExit):
        tls.require_bindable_ip(bad_host, 30)


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "example.com"])
def test_require_bindable_ip_noop_without_tls(host):
    """No TLS (days=0) → host is never constrained."""
    assert tls.require_bindable_ip(host, 0) is None


def test_context_from_pem_serves_verified_tls():
    """The /dev/fd pipe loader yields a context that serves a verifiable cert."""
    cert_pem, key_pem = tls._self_signed_pem(["localhost"], 30)
    ctx = tls._context_from_pem(cert_pem, key_pem)
    port = _serve_once(ctx)
    assert _client_recv(port, cadata=cert_pem.decode()) == b"OK"


def test_context_pins_tls13_and_disables_0rtt():
    """Floor is TLS 1.3 and session tickets (0-RTT PSK source) are off."""
    cert_pem, key_pem = tls._self_signed_pem(["localhost"], 30)
    ctx = tls._context_from_pem(cert_pem, key_pem)
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3
    assert ctx.options & ssl.OP_NO_TICKET


def test_negotiated_version_is_tls13():
    """A live handshake must land on TLS 1.3."""
    ctx = tls.build_ephemeral_context(host="127.0.0.1", days=30)
    port = _serve_once(ctx)
    cctx = ssl.create_default_context()
    cctx.check_hostname = False
    cctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with cctx.wrap_socket(raw, server_hostname="localhost") as t:
            assert t.version() == "TLSv1.3"


def test_no_cert_material_left_on_disk():
    """The zero-disk invariant: minting a context leaves no temp artifact."""
    before = set(_no_tls_tempdirs())
    ctx = tls.build_ephemeral_context(host="127.0.0.1", days=30)
    assert isinstance(ctx, ssl.SSLContext)
    after = set(_no_tls_tempdirs())
    assert after == before, f"cert material left on disk: {after - before}"
