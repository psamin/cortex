"""Uploads through InsForge's S3-compatible gateway (`/storage/v1/s3`) for files above the REST cap
(ACT checkpoints are about 200 MB; REST takes 50 MB by default, 200 MB at most).

A small S3 client over httpx: AWS Signature V4 and the multipart-upload calls, path-style only
(InsForge rejects virtual-hosted URLs). Keys are long-lived project S3 access keys created in the
InsForge dashboard (Storage -> Settings -> S3 Configuration); they never reach the browser.
Written against the S3 API so the backend needs no boto3 dependency.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

import httpx

PART_BYTES = 16 * 1024 * 1024  # S3 parts must be >= 5 MiB except the last
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class S3Error(RuntimeError):
    pass


def _uri(s: str, safe: str = "-_.~") -> str:
    return quote(s, safe=safe)


def sign(method: str, url: str, headers: dict[str, str], payload_sha256: str, key_id: str, secret: str,
         region: str, now: datetime | None = None, service: str = "s3") -> dict[str, str]:
    """`headers` plus x-amz-date, x-amz-content-sha256 and the SigV4 Authorization header.
    The Host header is signed as httpx will send it (host[:port])."""
    u = httpx.URL(url)
    amz_date = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    out = {**headers, "x-amz-date": amz_date, "x-amz-content-sha256": payload_sha256}
    canon = {k.lower(): re.sub(r"\s+", " ", str(v).strip()) for k, v in out.items()}
    canon["host"] = u.host + (f":{u.port}" if u.port else "")
    names = sorted(canon)
    query = sorted((_uri(k), _uri(v)) for k, v in u.params.multi_items())
    request = "\n".join([
        method.upper(), _uri(u.path, safe="/-_.~"), "&".join(f"{k}={v}" for k, v in query),
        "".join(f"{n}:{canon[n]}\n" for n in names), ";".join(names), payload_sha256])
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(request.encode()).hexdigest()])
    k = f"AWS4{secret}".encode()
    for part in (amz_date[:8], region, service, "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    out["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, SignedHeaders={';'.join(names)}, "
                            f"Signature={sig}")
    return out


class S3Gateway:
    def __init__(self, endpoint: str, region: str, key_id: str, secret: str,
                 transport: httpx.AsyncBaseTransport | None = None, part_bytes: int = PART_BYTES):
        self.endpoint = endpoint.rstrip("/")
        self.region, self.key_id, self.secret = region, key_id, secret
        self.part_bytes = part_bytes
        self.http = httpx.AsyncClient(timeout=300, transport=transport)

    def _url(self, bucket: str, key: str) -> str:
        return f"{self.endpoint}/{_uri(bucket)}/{_uri(key, safe='/-_.~')}"

    async def _req(self, method: str, url: str, body: bytes = b"", params: dict[str, str] | None = None,
                   headers: dict[str, str] | None = None) -> httpx.Response:
        full = str(httpx.URL(url, params=params or {}))
        signed = sign(method, full, headers or {}, hashlib.sha256(body).hexdigest(), self.key_id, self.secret,
                      self.region)
        r = await self.http.request(method, full, content=body, headers=signed)
        if r.status_code >= 300:
            raise S3Error(f"S3 {method} {httpx.URL(full).path} -> {r.status_code}: {r.text[:300]}")
        return r

    async def put_file(self, bucket: str, key: str, path: Path, content_type: str) -> str:
        """Upload `path` as bucket/key: one PutObject when it fits in a part, else multipart
        (aborted on any failure, so no orphaned parts are left behind)."""
        url = self._url(bucket, key)
        if path.stat().st_size <= self.part_bytes:
            await self._req("PUT", url, path.read_bytes(), headers={"content-type": content_type})
            return key
        r = await self._req("POST", url, params={"uploads": ""}, headers={"content-type": content_type})
        m = re.search(r"<UploadId>([^<]+)</UploadId>", r.text)
        if not m:
            raise S3Error(f"CreateMultipartUpload returned no UploadId: {r.text[:200]}")
        upload_id = m.group(1)
        try:
            etags: list[str] = []
            with path.open("rb") as f:
                while chunk := f.read(self.part_bytes):
                    r = await self._req("PUT", url, chunk,
                                        params={"partNumber": str(len(etags) + 1), "uploadId": upload_id})
                    etags.append(r.headers["etag"])
            done = "".join(f"<Part><PartNumber>{i}</PartNumber><ETag>{escape(e)}</ETag></Part>"
                           for i, e in enumerate(etags, 1))
            r = await self._req("POST", url, f"<CompleteMultipartUpload>{done}</CompleteMultipartUpload>".encode(),
                                params={"uploadId": upload_id}, headers={"content-type": "application/xml"})
            if "<Error>" in r.text:  # S3 can report a failed completion inside a 200 response
                raise S3Error(f"CompleteMultipartUpload failed: {r.text[:300]}")
        except BaseException:
            try:
                await self._req("DELETE", url, params={"uploadId": upload_id})
            except Exception:  # keep the original error
                pass
            raise
        return key
