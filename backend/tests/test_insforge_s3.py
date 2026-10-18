"""Large uploads through InsForge's S3-compatible gateway, against AWS's published SigV4 examples
and a strict in-memory fake gateway. No real InsForge project is contacted."""

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from urllib.parse import parse_qs

import httpx
import pytest
from test_insforge_contract import KEY, FakeInsForge

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Checkpoint
from cortex.services import training_service
from cortex.store.insforge import InsForgeBlobStore, InsForgeError
from cortex.store.s3 import S3Error, sign

AK, SK = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
BASE = "https://app.insforge.test"


@pytest.mark.parametrize("method,url,headers,payload,signature", [
    # docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html: the four example requests
    ("GET", "https://examplebucket.s3.amazonaws.com/test.txt", {"Range": "bytes=0-9"}, b"",
     "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"),
    ("PUT", "https://examplebucket.s3.amazonaws.com/test%24file.text",
     {"Date": "Fri, 24 May 2013 00:00:00 GMT", "x-amz-storage-class": "REDUCED_REDUNDANCY"},
     b"Welcome to Amazon S3.", "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"),
    ("GET", "https://examplebucket.s3.amazonaws.com/?lifecycle", {}, b"",
     "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543"),
    ("GET", "https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J", {}, b"",
     "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"),
])
def test_sigv4_matches_the_aws_examples(method, url, headers, payload, signature):
    out = sign(method, url, headers, hashlib.sha256(payload).hexdigest(), AK, SK, "us-east-1",
               now=datetime(2013, 5, 24, tzinfo=UTC))
    assert out["Authorization"].endswith(f"Signature={signature}")


class FakeS3:
    """Path-style S3 gateway at /storage/v1/s3 that re-checks every request's signature and payload hash."""

    def __init__(self, fail_part: int | None = None):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.aborted: list[str] = []
        self.calls: list[str] = []
        self.fail_part = fail_part

    def verify(self, req: httpx.Request) -> None:
        body = req.content
        assert req.headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
        auth = req.headers["authorization"]
        m = re.fullmatch(r"AWS4-HMAC-SHA256 Credential=([^/]+)/(\d{8})/([^/]+)/s3/aws4_request, "
                         r"SignedHeaders=([^,]+), Signature=([0-9a-f]{64})", auth)
        assert m and m.group(1) == AK, auth
        signed = m.group(4).split(";")
        assert {"host", "x-amz-date", "x-amz-content-sha256"} <= set(signed)
        extra = {h: req.headers[h] for h in signed if h not in ("host", "x-amz-date", "x-amz-content-sha256")}
        when = datetime.strptime(req.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        again = sign(req.method, str(req.url), extra, hashlib.sha256(body).hexdigest(), AK, SK, m.group(3), now=when)
        assert again["Authorization"] == auth, "signature does not match the request as sent"

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.verify(req)
        m = re.fullmatch(r"/storage/v1/s3/([^/]+)/(.+)", req.url.path)
        assert m, f"not path-style: {req.url}"
        obj, q = (m.group(1), m.group(2)), parse_qs(req.url.query.decode(), keep_blank_values=True)
        self.calls.append(f"{req.method} {'&'.join(sorted(q))}")
        if req.method == "POST" and "uploads" in q:
            uid = f"up{len(self.uploads)}"
            self.uploads[uid] = {}
            return httpx.Response(200, text=f"<InitiateMultipartUploadResult><UploadId>{uid}</UploadId>"
                                            "</InitiateMultipartUploadResult>")
        if req.method == "PUT" and "partNumber" in q:
            n = int(q["partNumber"][0])
            if n == self.fail_part:
                return httpx.Response(500, text="<Error><Code>InternalError</Code></Error>")
            self.uploads[q["uploadId"][0]][n] = req.content
            return httpx.Response(200, headers={"ETag": f'"{hashlib.md5(req.content).hexdigest()}"'})
        if req.method == "POST" and "uploadId" in q:
            parts = self.uploads.pop(q["uploadId"][0])
            listed = re.findall(r'<PartNumber>(\d+)</PartNumber><ETag>(?:"|&quot;)([0-9a-f]+)(?:"|&quot;)</ETag>',
                                req.content.decode())
            assert [int(n) for n, _ in listed] == sorted(parts)
            assert all(hashlib.md5(parts[int(n)]).hexdigest() == e for n, e in listed)
            self.objects[obj] = b"".join(parts[n] for n in sorted(parts))
            return httpx.Response(200, text="<CompleteMultipartUploadResult/>")
        if req.method == "DELETE" and "uploadId" in q:
            self.uploads.pop(q["uploadId"][0], None)
            self.aborted.append(q["uploadId"][0])
            return httpx.Response(204)
        if req.method == "PUT" and not q:
            self.objects[obj] = req.content
            return httpx.Response(200, headers={"ETag": '"x"'})
        return httpx.Response(501, text="<Error><Code>NotImplemented</Code></Error>")


def rest_with_s3_config(fake: FakeInsForge, region: str | None = "us-east-2"):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/storage/s3/config":
            return (httpx.Response(200, json={"endpoint": f"{BASE}/storage/v1/s3", "region": region}) if region
                    else httpx.Response(404, json={"error": "not found"}))
        return fake(req)
    return httpx.MockTransport(handler)


async def store(tmp_path, s3: FakeS3 | None, region="us-east-2", part_bytes=64 * 1024):
    rest = FakeInsForge()
    blobs = InsForgeBlobStore(BASE, KEY, "cortex-assets", transport=rest_with_s3_config(rest, region),
                              rest_max_bytes=100 * 1024, s3_keys=(AK, SK) if s3 else None,
                              s3_transport=httpx.MockTransport(s3) if s3 else None)
    await blobs.init()
    if blobs.s3:
        blobs.s3.part_bytes = part_bytes
    return blobs, rest


def blob(tmp_path, name: str, size: int):
    p = tmp_path / name
    p.write_bytes(os.urandom(size))
    return p


async def test_large_file_goes_through_s3_multipart(tmp_path):
    s3 = FakeS3()
    blobs, rest = await store(tmp_path, s3)
    p = blob(tmp_path, "model.safetensors", 300 * 1024)  # > REST cap -> 5 parts of 64 KiB
    key = "projects/prj_1/training/tj_1/pretrained_model/model.safetensors"
    await blobs.put_file(key, p, "application/octet-stream")
    assert s3.objects[("cortex-assets", key)] == p.read_bytes()
    assert s3.calls.count("PUT partNumber&uploadId") == 5 and not s3.aborted
    assert ("cortex-assets", key) not in rest.objects


async def test_parts_stay_under_the_gateway_limit(tmp_path):
    """A 16 MiB UploadPart is rejected and a ~30 MB PutObject drops. Parts are 5 MiB (#193)."""
    from cortex.store.s3 import PART_BYTES, S3Gateway

    assert PART_BYTES == 5 * 1024 * 1024
    s3 = FakeS3()
    gw = S3Gateway(f"{BASE}/storage/v1/s3", "us-east-2", AK, SK, transport=httpx.MockTransport(s3))
    p = blob(tmp_path, "ckpt.bin", 12 * 1024 * 1024)  # previously one PutObject, between the 10 MB OK and 30 MB drop
    await gw.put_file("cortex-assets", "ckpt.bin", p, "application/octet-stream")
    assert s3.calls.count("PUT partNumber&uploadId") == 3  # 5 + 5 + 2 MiB, never a 12 or 16 MiB body
    assert s3.objects[("cortex-assets", "ckpt.bin")] == p.read_bytes()


async def test_failed_part_aborts_the_upload(tmp_path):
    s3 = FakeS3(fail_part=3)
    blobs, _ = await store(tmp_path, s3)
    with pytest.raises(S3Error, match="500"):
        await blobs.put_file("big.bin", blob(tmp_path, "big.bin", 300 * 1024), "application/octet-stream")
    assert s3.aborted == ["up0"] and not s3.objects and not s3.uploads


async def test_small_files_still_use_rest(tmp_path):
    s3 = FakeS3()
    blobs, rest = await store(tmp_path, s3)
    p = blob(tmp_path, "config.json", 2 * 1024)
    await blobs.put_file("cfg/config.json", p, "application/json")
    assert rest.objects[("cortex-assets", "cfg/config.json")] == p.read_bytes() and not s3.calls


async def test_large_file_without_s3_keys_says_what_to_set(tmp_path):
    blobs, _ = await store(tmp_path, None)
    with pytest.raises(InsForgeError, match="INSFORGE_S3_ACCESS_KEY_ID"):
        await blobs.put_file("big.bin", blob(tmp_path, "big.bin", 300 * 1024), "application/octet-stream")


async def test_unknown_region_disables_s3_with_a_reason(tmp_path):
    blobs, _ = await store(tmp_path, FakeS3(), region=None)
    assert blobs.s3 is None
    with pytest.raises(InsForgeError, match="INSFORGE_S3_REGION"):
        await blobs.put_file("big.bin", blob(tmp_path, "big.bin", 300 * 1024), "application/octet-stream")


async def test_registered_checkpoints_reach_durable_storage(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path / "data", _env_file=None))
    s3 = FakeS3()
    ctx.blobs, rest = await store(tmp_path, s3)
    ck_dir = ctx.local("projects/prj_1/training/tj_1/pretrained_model")
    ck_dir.mkdir(parents=True)
    (ck_dir / "config.json").write_text(json.dumps({"type": "act"}))
    weights = blob(ck_dir, "model.safetensors", 300 * 1024)

    ck = await training_service.register_checkpoint(ctx, "prj_1", None, "projects/prj_1/training/tj_1/pretrained_model")
    assert (await ctx.repo.get("checkpoints", ck.id, Checkpoint)).policy_type == "act"
    prefix = "projects/prj_1/training/tj_1/pretrained_model"
    assert s3.objects[("cortex-assets", f"{prefix}/model.safetensors")] == weights.read_bytes()
    assert ("cortex-assets", f"{prefix}/config.json") in rest.objects
    audit = await ctx.store.list("tool_audit_logs", project_id="prj_1")
    assert "2 files stored" in audit[-1]["data"]["summary"]


async def test_checkpoint_stays_registered_when_the_upload_fails(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path / "data", _env_file=None))
    ctx.blobs, _ = await store(tmp_path, None)  # no S3 keys: the weights can't be uploaded
    ck_dir = ctx.local("projects/prj_1/ck")
    ck_dir.mkdir(parents=True)
    (ck_dir / "config.json").write_text(json.dumps({"type": "act"}))
    blob(ck_dir, "model.safetensors", 300 * 1024)
    ck = await training_service.register_checkpoint(ctx, "prj_1", None, "projects/prj_1/ck")
    assert await ctx.repo.get("checkpoints", ck.id, Checkpoint)
    audit = await ctx.store.list("tool_audit_logs", project_id="prj_1")
    assert "kept locally only, upload failed" in audit[-1]["data"]["summary"]
