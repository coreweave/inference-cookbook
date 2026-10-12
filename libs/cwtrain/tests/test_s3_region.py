# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""s3_client's region: required, from the caller or the AWS configuration, with no default."""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest

from cwtrain.errors import CWTrainError
from cwtrain.infra.storage import s3_client


@pytest.fixture
def aws_config(tmp_path: Path, monkeypatch) -> Path:
    """An empty AWS configuration, so the developer's own ~/.aws and environment cannot leak in."""
    config = tmp_path / "config"
    config.write_text("")
    credentials = tmp_path / "credentials"
    credentials.write_text("[default]\naws_access_key_id = k\naws_secret_access_key = s\n")
    for var in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "CW_S3_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")  # a profile without keys: skip IMDS
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    return config


def test_no_region_anywhere_is_an_error(aws_config: Path) -> None:
    with pytest.raises(CWTrainError, match="Availability Zone") as raised:
        s3_client()
    for source in ("AWS_REGION", "AWS_DEFAULT_REGION", "profile's region"):
        assert source in str(raised.value)


def test_a_blank_aws_region_does_not_count(aws_config: Path, monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", " ")
    with pytest.raises(CWTrainError, match="no object storage region"):
        s3_client()


@pytest.mark.parametrize("var", ["AWS_REGION", "AWS_DEFAULT_REGION"])
def test_region_from_the_environment(aws_config: Path, monkeypatch, var: str) -> None:
    monkeypatch.setenv(var, "US-WEST-04A")
    assert s3_client().meta.region_name == "US-WEST-04A"


def test_a_padded_region_from_the_aws_configuration_is_trimmed(aws_config, monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", " US-WEST-04A ")
    assert s3_client().meta.region_name == "US-WEST-04A"


def test_region_from_the_profile(aws_config: Path, monkeypatch) -> None:
    aws_config.write_text("[profile cw]\nregion = US-CENTRAL-05A\n")
    monkeypatch.setenv("AWS_PROFILE", "cw")
    assert s3_client().meta.region_name == "US-CENTRAL-05A"


def test_aws_region_then_aws_default_region_then_the_profile(aws_config: Path, monkeypatch) -> None:
    aws_config.write_text("[default]\nregion = US-CENTRAL-05A\n")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "EU-SOUTH-04A")
    monkeypatch.setenv("AWS_REGION", "US-WEST-04A")
    assert s3_client().meta.region_name == "US-WEST-04A"
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    assert s3_client().meta.region_name == "EU-SOUTH-04A"
    monkeypatch.delenv("AWS_DEFAULT_REGION")
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    assert s3_client().meta.region_name == "US-CENTRAL-05A"


def test_an_explicit_region_wins(aws_config: Path, monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "US-WEST-04A")
    assert s3_client(region_name="RNO2A").meta.region_name == "RNO2A"
