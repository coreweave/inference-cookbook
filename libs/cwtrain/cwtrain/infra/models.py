# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Typed views of the Dedicated Inference API objects: gateways, deployments and hot-loads.

The API is protobuf transcoded to JSON. Enums arrive as their proto names
(``"HOT_LOAD_STATE_COMPLETED"``), but some tools send the integer, so every enum here parses both.
Timestamps arrive as RFC 3339 strings and are parsed to :class:`datetime`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any, TypeVar

log = logging.getLogger(__name__)

E = TypeVar("E", bound="ProtoEnum")


class ProtoEnum(IntEnum):
    """An enum that round-trips the protobuf JSON representation.

    Subclasses register their ``ENUM_NAME_`` wire prefix in :data:`_PREFIXES`, because a plain
    class attribute in an Enum body becomes a member.
    """

    @classmethod
    def parse(cls: type[E], value: Any) -> E:
        """Parse a proto name (with or without prefix) or integer. Unknown values become 0."""
        if value is None:
            return cls(0)
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                return cls(0)
        if isinstance(value, str):
            name = value.upper()
            prefix = cls._prefix_value()
            if prefix and name.startswith(prefix):
                name = name[len(prefix) :]
            try:
                return cls[name]
            except KeyError:
                if name.isdigit():
                    return cls.parse(int(name))
                return cls(0)
        raise TypeError(f"cannot parse {value!r} as {cls.__name__}")

    @classmethod
    def _prefix_value(cls) -> str:
        return _PREFIXES.get(cls, "")

    @property
    def wire(self) -> str:
        """The proto JSON name, for request bodies."""
        return f"{self._prefix_value()}{self.name}"


class SnapshotType(ProtoEnum):
    UNSPECIFIED = 0
    FULL = 1
    INCREMENTAL = 2


class CompressionFormat(ProtoEnum):
    UNSPECIFIED = 0
    ZSTD = 1


class ChecksumFormat(ProtoEnum):
    UNSPECIFIED = 0
    ADLER32 = 1


class PromptCachePolicy(ProtoEnum):
    UNSPECIFIED = 0
    RESET_ALL = 1
    PRESERVE = 2


class HotLoadState(ProtoEnum):
    UNSPECIFIED = 0
    PENDING = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4
    CANCELED = 5

    @property
    def terminal(self) -> bool:
        return self in (HotLoadState.COMPLETED, HotLoadState.FAILED, HotLoadState.CANCELED)


class ReplicaState(ProtoEnum):
    UNSPECIFIED = 0
    PENDING_UPDATE = 1
    UPDATING = 2
    READY = 3
    ERROR = 4


class TransitionMode(ProtoEnum):
    UNSPECIFIED = 0
    ASYNC = 1


class ResourceStatus(ProtoEnum):
    """The lifecycle status gateways and deployments share (the API's ``Status``)."""

    UNSPECIFIED = 0
    CREATING = 1
    UPDATING = 2
    DELETING = 3
    ERROR = 4
    FAILED = 5
    READY = 6


class ConditionStatus(ProtoEnum):
    UNSPECIFIED = 0
    TRUE = 1
    FALSE = 2
    UNKNOWN = 3


_PREFIXES: dict[type[ProtoEnum], str] = {
    SnapshotType: "SNAPSHOT_TYPE_",
    CompressionFormat: "COMPRESSION_FORMAT_",
    ChecksumFormat: "CHECKSUM_FORMAT_",
    PromptCachePolicy: "PROMPT_CACHE_POLICY_",
    HotLoadState: "HOT_LOAD_STATE_",
    ReplicaState: "",  # proto values carry no prefix (buf:lint:ignore ENUM_VALUE_PREFIX)
    TransitionMode: "TRANSITION_MODE_",
    ResourceStatus: "STATUS_",
    ConditionStatus: "STATUS_",  # Condition.status uses the same STATUS_ prefix on the wire
}


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Normalize fractional seconds to exactly six digits (microseconds).
    if "." in text:
        head, _, tail = text.partition(".")
        frac = ""
        while tail and tail[0].isdigit():
            frac += tail[0]
            tail = tail[1:]
        text = f"{head}.{frac[:6].ljust(6, '0')}{tail}"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        log.warning("ignoring unparseable timestamp %r", value)
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@dataclass(frozen=True)
class IncrementalSnapshotMetadata:
    previous_snapshot_identity: str
    compression_format: CompressionFormat = CompressionFormat.ZSTD
    checksum_format: ChecksumFormat = ChecksumFormat.ADLER32

    def to_json(self) -> dict[str, Any]:
        return {
            "previousSnapshotIdentity": self.previous_snapshot_identity,
            "compressionFormat": self.compression_format.wire,
            "checksumFormat": self.checksum_format.wire,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> IncrementalSnapshotMetadata | None:
        if not data:
            return None
        return cls(
            previous_snapshot_identity=data.get("previousSnapshotIdentity", ""),
            compression_format=CompressionFormat.parse(data.get("compressionFormat")),
            checksum_format=ChecksumFormat.parse(data.get("checksumFormat")),
        )


@dataclass(frozen=True)
class SnapshotRef:
    identity: str
    type: SnapshotType
    hot_load_id: str = ""
    compression_format: CompressionFormat = CompressionFormat.UNSPECIFIED
    checksum_format: ChecksumFormat = ChecksumFormat.UNSPECIFIED

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SnapshotRef:
        return cls(
            identity=data.get("identity", ""),
            type=SnapshotType.parse(data.get("type")),
            hot_load_id=data.get("hotLoadId", ""),
            compression_format=CompressionFormat.parse(data.get("compressionFormat")),
            checksum_format=ChecksumFormat.parse(data.get("checksumFormat")),
        )


@dataclass(frozen=True)
class Condition:
    type: str
    status: ConditionStatus
    reason: str = ""
    message: str = ""
    zone: str = ""
    last_update_time: datetime | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Condition:
        return cls(
            type=data.get("type", ""),
            status=ConditionStatus.parse(data.get("status")),
            reason=data.get("reason", ""),
            message=data.get("message", ""),
            zone=data.get("zone", ""),
            last_update_time=parse_time(data.get("lastUpdateTime")),
        )


@dataclass(frozen=True)
class HotLoadSpec:
    id: str
    deployment_id: str
    identity: str
    type: SnapshotType
    organization_id: str = ""
    incremental_snapshot_metadata: IncrementalSnapshotMetadata | None = None
    prompt_cache_policy: PromptCachePolicy = PromptCachePolicy.UNSPECIFIED
    snapshot_chain: tuple[SnapshotRef, ...] = ()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HotLoadSpec:
        return cls(
            id=data.get("id", ""),
            deployment_id=data.get("deploymentId", ""),
            identity=data.get("identity", ""),
            type=SnapshotType.parse(data.get("type")),
            organization_id=data.get("organizationId", ""),
            incremental_snapshot_metadata=IncrementalSnapshotMetadata.from_json(
                data.get("incrementalSnapshotMetadata")
            ),
            prompt_cache_policy=PromptCachePolicy.parse(data.get("promptCachePolicy")),
            snapshot_chain=tuple(SnapshotRef.from_json(r) for r in data.get("snapshotChain") or []),
        )


@dataclass(frozen=True)
class HotLoadStatus:
    state: HotLoadState
    created_at: datetime | None = None
    updated_at: datetime | None = None
    finished_at: datetime | None = None
    canceled_at: datetime | None = None
    conditions: tuple[Condition, ...] = ()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HotLoadStatus:
        return cls(
            state=HotLoadState.parse(data.get("state")),
            created_at=parse_time(data.get("createdAt")),
            updated_at=parse_time(data.get("updatedAt")),
            finished_at=parse_time(data.get("finishedAt")),
            canceled_at=parse_time(data.get("canceledAt")),
            conditions=tuple(Condition.from_json(c) for c in data.get("conditions") or []),
        )


@dataclass(frozen=True)
class HotLoad:
    spec: HotLoadSpec
    status: HotLoadStatus

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HotLoad:
        return cls(
            spec=HotLoadSpec.from_json(data.get("spec") or {}),
            status=HotLoadStatus.from_json(data.get("status") or {}),
        )

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def identity(self) -> str:
        return self.spec.identity

    @property
    def deployment_id(self) -> str:
        return self.spec.deployment_id

    @property
    def state(self) -> HotLoadState:
        return self.status.state

    @property
    def terminal(self) -> bool:
        return self.status.state.terminal

    @property
    def completed(self) -> bool:
        return self.status.state is HotLoadState.COMPLETED


@dataclass(frozen=True)
class HotLoadReplica:
    name: str
    state: ReplicaState
    zone: str = ""
    current_snapshot_identity: str = ""
    error_reason: str = ""
    transitioned_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HotLoadReplica:
        return cls(
            name=data.get("name", ""),
            state=ReplicaState.parse(data.get("state")),
            zone=data.get("zone", ""),
            current_snapshot_identity=data.get("currentSnapshotIdentity", ""),
            error_reason=data.get("errorReason", ""),
            transitioned_at=parse_time(data.get("transitionedAt")),
            started_at=parse_time(data.get("startedAt")),
            finished_at=parse_time(data.get("finishedAt")),
            updated_at=parse_time(data.get("updatedAt")),
        )


@dataclass(frozen=True)
class DeploymentHotLoadConfig:
    """The immutable ``spec.hot_load`` section of a deployment."""

    bucket: str
    path_prefix: str = ""
    transition_mode: TransitionMode = TransitionMode.UNSPECIFIED

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> DeploymentHotLoadConfig | None:
        if not data:
            return None
        return cls(
            bucket=data.get("bucket", ""),
            path_prefix=data.get("pathPrefix", ""),
            transition_mode=TransitionMode.parse(data.get("transitionMode")),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"bucket": self.bucket, "transitionMode": self.transition_mode.wire}
        if self.path_prefix:
            out["pathPrefix"] = self.path_prefix
        return out


@dataclass(frozen=True)
class Gateway:
    """Dedicated Inference gateway identity, readiness and public endpoints."""

    id: str
    name: str
    status: ResourceStatus
    organization_id: str = ""
    zones: tuple[str, ...] = ()
    endpoints: tuple[str, ...] = ()
    conditions: tuple[Condition, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Gateway:
        spec = data.get("spec") or {}
        status = data.get("status") or {}
        return cls(
            id=spec.get("id", ""),
            name=spec.get("name", ""),
            status=ResourceStatus.parse(status.get("status")),
            organization_id=spec.get("organizationId", ""),
            zones=tuple(spec.get("zones") or []),
            endpoints=tuple(status.get("endpoints") or []),
            conditions=tuple(Condition.from_json(c) for c in status.get("conditions") or []),
            raw=data,
        )

    @property
    def ready(self) -> bool:
        return self.status is ResourceStatus.READY


@dataclass(frozen=True)
class Deployment:
    """The subset of a deployment that training against it needs; the rest is in ``raw``."""

    id: str
    name: str
    status: ResourceStatus
    model_name: str = ""
    model_bucket: str = ""
    model_path: str = ""
    engine: str = ""
    hot_load: DeploymentHotLoadConfig | None = None
    gateway_ids: tuple[str, ...] = ()
    conditions: tuple[Condition, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Deployment:
        spec = data.get("spec") or {}
        status = data.get("status") or {}
        model = spec.get("model") or {}
        runtime = spec.get("runtime") or {}
        return cls(
            id=spec.get("id", ""),
            name=spec.get("name", ""),
            status=ResourceStatus.parse(status.get("status")),
            model_name=model.get("name", ""),
            model_bucket=model.get("bucket", ""),
            model_path=model.get("path", ""),
            engine=runtime.get("engine", ""),
            hot_load=DeploymentHotLoadConfig.from_json(spec.get("hotLoad")),
            gateway_ids=tuple(spec.get("gatewayIds") or []),
            conditions=tuple(Condition.from_json(c) for c in status.get("conditions") or []),
            raw=data,
        )

    @property
    def ready(self) -> bool:
        return self.status is ResourceStatus.READY

    @property
    def hot_load_enabled(self) -> bool:
        return self.hot_load is not None
