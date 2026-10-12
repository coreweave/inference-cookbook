# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Which object-storage endpoint s3_client picks from arguments, CW_S3_ENDPOINT and AWS config."""

from __future__ import annotations

from pathlib import Path

import boto3
import botocore.session
import pytest
from botocore.config import Config
from botocore.exceptions import InvalidConfigError

from cwtrain.infra.storage import DEFAULT_S3_ENDPOINT, s3_client

LOTA = "http://cwlota.com"


@pytest.fixture
def aws_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty AWS configuration, so the developer's own ~/.aws and environment do not leak in."""
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS",
        "CW_S3_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config"
    config.write_text("")
    (tmp_path / "credentials").write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)  # the default session caches the config
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "US-EAST-04A")  # required; these tests are about the endpoint
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")  # no credentials: don't wait on IMDS
    return config


def test_nothing_configured_uses_the_primary_endpoint(aws_config: Path) -> None:
    assert s3_client().meta.endpoint_url == DEFAULT_S3_ENDPOINT


def test_the_endpoint_check_resolves_no_credentials(aws_config: Path, monkeypatch) -> None:
    # Resolving credentials can stall on an instance metadata timeout. Only the client needs
    # them, so the credential chain runs once.
    lookups = []
    get_credentials = botocore.session.Session.get_credentials

    def counting(self):
        lookups.append(1)
        return get_credentials(self)

    monkeypatch.setattr(botocore.session.Session, "get_credentials", counting)
    s3_client()
    assert len(lookups) == 1


def test_the_callers_config_is_left_as_given(aws_config: Path) -> None:
    # botocore rewrites a config's retries in place (max_attempts -> total_max_attempts)
    config = Config(retries={"max_attempts": 3}, ignore_configured_endpoint_urls=True)
    client = s3_client(config=config)
    assert config.retries == {"max_attempts": 3}
    assert client.meta.config.retries["total_max_attempts"] == 4  # still the caller's settings


@pytest.mark.parametrize("name", ["AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL"])
def test_standard_aws_endpoint_variables_are_honored(aws_config, monkeypatch, name) -> None:
    monkeypatch.setenv(name, LOTA)
    assert s3_client().meta.endpoint_url == LOTA


@pytest.mark.parametrize(
    ("region", "name", "endpoint"),
    [
        ("us-west-2", "AWS_ENDPOINT_URL_S3", "https://s3.us-west-2.amazonaws.com"),
        ("us-east-1", "AWS_ENDPOINT_URL", "https://s3.amazonaws.com"),
        ("US-EAST-04A", "AWS_ENDPOINT_URL_S3", "https://s3.us-west-2.amazonaws.com"),
    ],
)
def test_a_configured_aws_endpoint_is_honored(aws_config, monkeypatch, region, name, endpoint):
    # storage is CoreWeave's by default, not exclusively: an explicit AWS endpoint stays, even the
    # one boto3 would pick for the region anyway
    monkeypatch.setenv("AWS_REGION", region)
    monkeypatch.setenv(name, endpoint)
    assert s3_client().meta.endpoint_url == endpoint


@pytest.mark.parametrize("endpoint", [LOTA, "https://s3.eu-west-1.amazonaws.com"])
def test_a_profile_endpoint_is_honored(aws_config: Path, monkeypatch, endpoint: str) -> None:
    aws_config.write_text(f"[profile cw]\nendpoint_url = {endpoint}\n")
    monkeypatch.setenv("AWS_PROFILE", "cw")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert s3_client().meta.endpoint_url == endpoint


def test_a_services_section_endpoint_is_honored(aws_config: Path, monkeypatch) -> None:
    aws_config.write_text(
        f"[profile cw]\nservices = storage\n\n[services storage]\ns3 =\n  endpoint_url = {LOTA}\n"
    )
    monkeypatch.setenv("AWS_PROFILE", "cw")
    assert s3_client().meta.endpoint_url == LOTA


def test_a_missing_services_section_is_an_error(aws_config: Path, monkeypatch) -> None:
    # boto3 refuses it; quietly using the CoreWeave endpoint would hide the mistake
    aws_config.write_text("[profile cw]\nservices = storage\n")
    monkeypatch.setenv("AWS_PROFILE", "cw")
    with pytest.raises(InvalidConfigError):
        s3_client()


@pytest.mark.parametrize("where", ["environment", "profile", "client config"])
def test_ignored_configured_endpoints_give_the_primary_endpoint(aws_config, monkeypatch, where):
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", LOTA)
    kwargs = {}
    if where == "environment":
        monkeypatch.setenv("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "true")
    elif where == "profile":
        aws_config.write_text("[profile cw]\nignore_configured_endpoint_urls = true\n")
        monkeypatch.setenv("AWS_PROFILE", "cw")
    else:
        kwargs["config"] = Config(ignore_configured_endpoint_urls=True)
    assert s3_client(**kwargs).meta.endpoint_url == DEFAULT_S3_ENDPOINT


def test_cw_s3_endpoint_overrides_the_aws_configuration(aws_config, monkeypatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://other.example")
    monkeypatch.setenv("CW_S3_ENDPOINT", LOTA)
    assert s3_client().meta.endpoint_url == LOTA
    # a blank value, which an unset run.env entry renders as, does not count
    monkeypatch.setenv("CW_S3_ENDPOINT", " ")
    assert s3_client().meta.endpoint_url == "https://other.example"


def test_an_explicit_endpoint_wins(aws_config, monkeypatch) -> None:
    monkeypatch.setenv("CW_S3_ENDPOINT", LOTA)
    assert s3_client("https://explicit.example").meta.endpoint_url == "https://explicit.example"
