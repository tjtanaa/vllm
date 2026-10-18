# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    MoRIIOTransferAck,
    TransferError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorWorker,
    get_moriio_expected_ack_count,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
    validate_moriio_heterogeneous_tp_kv_heads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)


def test_remote_tp_rank_same_tp_maps_to_self():
    assert [get_moriio_remote_tp_rank(rank, 4, 4) for rank in range(4)] == [
        0,
        1,
        2,
        3,
    ]


def test_remote_tp_rank_p4_d8_floor_maps_decode_to_prefill():
    assert [get_moriio_remote_tp_rank(rank, 8, 4) for rank in range(8)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_remote_tp_rank_p8_d4_maps_to_first_prefill_rank_per_pair():
    assert [get_moriio_remote_tp_rank(rank, 4, 8) for rank in range(4)] == [
        0,
        2,
        4,
        6,
    ]


@pytest.mark.parametrize(
    ("local_tp_rank", "local_tp_size", "remote_tp_size"),
    [
        (0, 6, 4),
        (0, 4, 6),
    ],
)
def test_remote_tp_rank_invalid_non_multiple_tp_raises(
    local_tp_rank: int, local_tp_size: int, remote_tp_size: int
):
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_remote_tp_rank(local_tp_rank, local_tp_size, remote_tp_size)


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 4, 8),
        (8, 4, 4),
        (4, 8, 4),
    ],
)
def test_heterogeneous_tp_head_guard_allows_supported_layouts(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size,
        remote_tp_size,
        total_num_kv_heads,
        is_mla=False,
    )


def test_heterogeneous_tp_head_guard_allows_mla_layouts():
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size=2,
        remote_tp_size=4,
        total_num_kv_heads=4,
        is_mla=True,
    )


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 2, 4),
        (2, 4, 4),
    ],
)
def test_heterogeneous_tp_head_guard_rejects_split_kv_heads(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    with pytest.raises(NotImplementedError, match="replicated KV heads"):
        validate_moriio_heterogeneous_tp_kv_heads(
            local_tp_size,
            remote_tp_size,
            total_num_kv_heads,
            is_mla=False,
        )


def test_expected_ack_count_for_homogeneous_or_smaller_consumer_tp_is_one():
    assert get_moriio_expected_ack_count(4, 4) == 1
    assert get_moriio_expected_ack_count(8, 4) == 1


def test_expected_ack_count_for_decode_fan_in():
    assert get_moriio_expected_ack_count(4, 8) == 2


def test_expected_ack_count_rejects_non_multiple_fan_in():
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_expected_ack_count(4, 6)


def test_plain_string_ack_is_backward_compatible_single_ack():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            "tx-plain",
            producer_tp_size=4,
            live_transfer_ids={"tx-plain"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-plain"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-plain"}


def test_structured_release_ack_waits_for_all_expected_acks():
    ack = MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {"tx-fanin": 1}
    assert completed_transfer_ids == set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-fanin"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-fanin"}


def test_duplicate_ack_after_completion_does_not_resolve_twice():
    ack = MoRIIOTransferAck("tx-dup", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-dup"
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-dup"}


def test_ack_for_non_live_transfer_is_ignored():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            MoRIIOTransferAck("tx-stale", consumer_tp_size=8),
            producer_tp_size=4,
            live_transfer_ids={"tx-live"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == set()


def test_worker_get_finished_counts_structured_release_fan_in():
    class FakeWrapper:
        def __init__(self):
            self.batches = [
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
            ]

        def pop_finished_req_ids(self):
            return self.batches.pop(0)

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.world_size = 4
    worker.moriio_wrapper = FakeWrapper()
    worker.transfer_id_to_request_id = {"tx-fanin": "req-fanin"}
    worker._consumer_notification_counts = {}
    worker._completed_consumer_notifications = set()
    worker._pending_unmapped_acks = []

    assert worker.get_finished() == (set(), set())
    assert worker._consumer_notification_counts == {"tx-fanin": 1}

    assert worker.get_finished() == ({"req-fanin"}, set())
    assert worker._consumer_notification_counts == {}
    assert worker._completed_consumer_notifications == {"tx-fanin"}


@pytest.mark.parametrize("exit_kind", ["killed", "eof"])
def test_worker_reports_unexpected_heartbeat_exit(exit_kind):
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_heartbeat import (
        MoRIIOHeartbeat,
    )

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.moriio_wrapper = SimpleNamespace(
        pop_finished_req_ids=lambda: [], shutdown=lambda: None
    )
    worker._pending_unmapped_acks = []
    worker._heartbeat = MoRIIOHeartbeat("tcp://127.0.0.1:1", {"type": "P"}, 0.02)
    process = worker._heartbeat._process
    try:
        assert worker.get_finished() == (set(), set())
        if exit_kind == "killed":
            process.kill()
        else:
            process.stdin.close()
        returncode = process.wait(timeout=5)
        with pytest.raises(RuntimeError, match=f"heartbeat.*{returncode}"):
            worker.get_finished()
    finally:
        worker.shutdown()


def test_read_completion_sends_structured_release_with_consumer_tp_size():
    class DoneStatus:
        def Succeeded(self):
            return True

        def Failed(self):
            return False

    class FakeWrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.sent = []

        # Exercise the real batch verdict rather than restating it here.
        poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

        def send_notify(
            self,
            transfer_id,
            host,
            port,
            message_type=None,
            message_fields=None,
        ):
            self.sent.append((transfer_id, host, port, message_type, message_fields))

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = FakeWrapper()
    # A layer maps to the list of reads posted for it (a KDA layer posts two).
    worker._recving_transfers = {"req": {"layer0": [DoneStatus(), DoneStatus()]}}
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", ("7000",), "tx-release")
    }
    # Transfer-timeout reaping state consulted by _pop_done_transfers.
    worker._recving_transfers_start = {}
    # Load-error bookkeeping cleared alongside a completed transfer.
    worker._recving_local_blocks = {}
    worker._invalid_block_ids = set()

    assert worker._pop_done_transfers() == {"tx-release"}
    assert worker.moriio_wrapper.sent == [
        (
            "tx-release",
            "127.0.0.1",
            "7000",
            "release",
            {"consumer_tp_size": 8},
        )
    ]
    assert worker._recving_transfers == {}
    assert worker._recving_transfers_callback_addr == {}


def test_aborted_read_without_local_blocks_only_releases_remote_blocks():
    class FakeWrapper:
        def __init__(self):
            self.sent = []

        def send_notify(
            self,
            transfer_id,
            host,
            port,
            message_type=None,
            message_fields=None,
        ):
            self.sent.append((transfer_id, host, port, message_type, message_fields))

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.mode = MoRIIOMode.READ
    worker.world_size = 8
    worker.moriio_wrapper = FakeWrapper()

    worker._read_blocks(
        local_block_ids=[],
        remote_block_ids=[[10, 11], [90]],
        dst_engine_id="prefill",
        request_id="req-aborted",
        transfer_id="tx-aborted",
        remote_host="127.0.0.1",
        remote_notify_port=7000,
        remote_tp_size=8,
        remote_dp_rank=0,
        chosen_tp=2,
    )

    assert worker.moriio_wrapper.sent == [
        (
            "tx-aborted",
            "127.0.0.1",
            "7002",
            "release",
            {"consumer_tp_size": 8},
        )
    ]


@pytest.mark.parametrize(
    ("decode_tp", "prefill_tp", "eff_tp", "expected_ranks"),
    [
        pytest.param(2, 2, 1, [1], id="equal-tp"),
        pytest.param(4, 2, 1, [1], id="decode-fan-out"),
        pytest.param(2, 4, 0, [0, 1], id="fan-in-first-group"),
        pytest.param(2, 4, 2, [2, 3], id="fan-in-second-group"),
        pytest.param(1, 4, 2, [0, 1, 2, 3], id="flexible-mirror"),
    ],
)
def test_read_release_ports_cover_every_prefill_rank_of_the_group(
    decode_tp, prefill_tp, eff_tp, expected_ranks
):
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = decode_tp

    ports = worker._read_release_ports(7000, 0, eff_tp, prefill_tp)

    assert ports == tuple(str(7000 + rank) for rank in expected_ranks)


class _ReleaseRecordingWrapper:
    def __init__(self):
        self.lock = threading.Lock()
        self.sent = []

    poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

    def send_notify(
        self, transfer_id, host, port, message_type=None, message_fields=None
    ):
        self.sent.append((transfer_id, port, message_type, message_fields))

    def shutdown(self):
        pass


class _DoneStatus:
    def Succeeded(self):
        return True

    def Failed(self):
        return False


@pytest.mark.parametrize(
    ("prefill_tp", "decode_tp"),
    [(2, 2), (1, 2), (2, 4), (2, 1), (4, 2), (4, 1), (8, 2)],
)
def test_completed_reads_release_every_prefill_rank_exactly_once(prefill_tp, decode_tp):
    """Prefill frees a request only once all its TP ranks report it sent.

    Each decode rank reads from one prefill rank and releases the ranks it
    stands for; every prefill rank must then resolve the transfer, or the
    request's blocks wait for the scheduler's defer_timeout.
    """
    sent = []
    for decode_rank in range(decode_tp):
        worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
        worker.world_size = decode_tp
        worker.tp_rank = decode_rank
        worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
        worker.moriio_wrapper = _ReleaseRecordingWrapper()
        worker._recving_transfers = {"req": {"layer0": [_DoneStatus()]}}
        eff_tp = worker._remote_tp_rank(prefill_tp)
        worker._recving_transfers_callback_addr = {
            "req": (
                "127.0.0.1",
                worker._read_release_ports(7000, 0, eff_tp, prefill_tp),
                "tx",
            )
        }
        worker._recving_transfers_start = {}
        worker._recving_local_blocks = {}
        worker._invalid_block_ids = set()

        assert worker._pop_done_transfers() == {"tx"}
        sent.extend(worker.moriio_wrapper.sent)

    for prefill_rank in range(prefill_tp):
        counts: dict[str, int] = {}
        completed: set[str] = set()
        resolved = [
            resolve_moriio_transfer_ack(
                MoRIIOTransferAck(transfer_id, fields["consumer_tp_size"]),
                producer_tp_size=prefill_tp,
                live_transfer_ids={"tx"},
                notification_counts=counts,
                completed_transfer_ids=completed,
            )
            for transfer_id, port, message_type, fields in sent
            if port == str(7000 + prefill_rank) and message_type == "release"
        ]
        assert [r for r in resolved if r is not None] == ["tx"], prefill_rank
        assert counts == {}, prefill_rank


def test_aborted_fan_in_read_releases_every_prefill_rank_of_the_group():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.mode = MoRIIOMode.READ
    worker.world_size = 2
    worker.tp_rank = 1
    worker.moriio_wrapper = _ReleaseRecordingWrapper()

    worker._read_blocks(
        local_block_ids=[],
        remote_block_ids=[[10, 11]],
        dst_engine_id="prefill",
        request_id="req-aborted",
        transfer_id="tx-aborted",
        remote_host="127.0.0.1",
        remote_notify_port=7000,
        remote_tp_size=4,
    )

    assert worker.moriio_wrapper.sent == [
        ("tx-aborted", "7002", "release", {"consumer_tp_size": 2}),
        ("tx-aborted", "7003", "release", {"consumer_tp_size": 2}),
    ]


def test_read_completion_waits_for_every_posted_read():
    class Status:
        def __init__(self, done: bool):
            self.done = done

        def Succeeded(self):
            return self.done

        def Failed(self):
            return False

    class FakeWrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.sent = []

        poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

        def send_notify(self, *args, **kwargs):
            self.sent.append((args, kwargs))

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = {
        "req": {"kda_layer": [Status(done=True), Status(done=False)]}
    }
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", ("7000",), "tx-pending")
    }
    worker._recving_transfers_start = {"req": float("inf")}
    worker._recving_local_blocks = {"req": [1, 2]}
    worker._invalid_block_ids = set()

    assert worker._pop_done_transfers() == set()
    assert "req" in worker._recving_transfers
    assert worker.moriio_wrapper.sent == []


@pytest.mark.parametrize(
    ("has_mamba", "expected_invalid"),
    [(False, {1, 2}), (True, set())],
)
def test_failed_read_reports_blocks_only_without_hma(has_mamba, expected_invalid):
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = has_mamba
    worker._recving_local_blocks = {"req": [1, 2]}
    worker._invalid_block_ids = set()

    worker._record_failed_recv("req")

    assert worker.get_block_ids_with_load_errors() == expected_invalid


def test_hybrid_step_barrier_fails_closed(monkeypatch):
    class FailingWrapper:
        def waiting_for_transfer_complete(self, _statuses):
            raise TransferError("failed")

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = True
    worker._reads_issued_this_step = [object()]
    worker._mamba_reads_this_step = [object()]
    worker.moriio_wrapper = FailingWrapper()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=None),
    )

    with pytest.raises(TransferError, match="failed"):
        worker._await_reads_issued_this_step()


def test_requested_cudagraph_mode_is_never_overridden():
    # The configured cudagraph mode is always honored: the barrier fires when
    # the operator sets cudagraph_mode=PIECEWISE, and READ mode with full
    # graphs only warns instead of silently forcing PIECEWISE.
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": True}) is False
    )
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": False}) is False
    )
