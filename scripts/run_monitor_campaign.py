"""Run the frozen formal monitor training campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.artifacts import canonical_json_bytes
from avalanche.monitors.campaign import (
    CampaignError,
    close_campaign,
    current_attempt_identity,
    execute_fitting_process,
    load_campaign,
    materialize_campaign,
    prepare_campaign,
    prepare_campaign_close,
)
from avalanche.monitors.releases import (
    RemoteAsset,
    RemoteRelease,
    publish_marker_release,
)

REPOSITORY = "antonstrover/Avalanche"


class GitHubReleaseTransport:
    """Publish campaign markers through the authenticated GitHub CLI."""

    def find_by_tag(self, tag: str) -> RemoteRelease | None:
        result = _gh(
            ("api", f"repos/{REPOSITORY}/releases/tags/{tag}"),
            check=False,
        )
        if result.returncode != 0 and "HTTP 404" in result.stderr:
            return None
        if result.returncode != 0:
            raise CampaignError("the release lookup failed")
        return _remote_release(json.loads(result.stdout))

    def get_release(self, release_id: str) -> RemoteRelease:
        result = _gh(("api", f"repos/{REPOSITORY}/releases/{release_id}"))
        return _remote_release(json.loads(result.stdout))

    def create_draft(self, tag: str, target_revision: str) -> RemoteRelease:
        result = _gh(
            (
                "api",
                "--method",
                "POST",
                f"repos/{REPOSITORY}/releases",
                "-f",
                f"tag_name={tag}",
                "-f",
                f"target_commitish={target_revision}",
                "-F",
                "draft=true",
                "-f",
                f"name={tag}",
            )
        )
        return _remote_release(json.loads(result.stdout))

    def upload_asset(self, release_id: str, name: str, content: bytes) -> None:
        release = self.get_release(release_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_bytes(content)
            _gh(("release", "upload", release.tag, str(path)))

    def download_asset(self, release_id: str, name: str) -> bytes:
        value = _gh_json(("api", f"repos/{REPOSITORY}/releases/{release_id}"))
        matches = [item for item in value["assets"] if item["name"] == name]
        if len(matches) != 1:
            raise CampaignError("the release asset identity is not unique")
        result = _gh(
            (
                "api",
                "-H",
                "Accept: application/octet-stream",
                f"repos/{REPOSITORY}/releases/assets/{matches[0]['id']}",
            ),
            text=False,
        )
        return result.stdout

    def publish(self, release_id: str) -> RemoteRelease:
        result = _gh(
            (
                "api",
                "--method",
                "PATCH",
                f"repos/{REPOSITORY}/releases/{release_id}",
                "-F",
                "draft=false",
            )
        )
        return _remote_release(json.loads(result.stdout))


def build_parser() -> argparse.ArgumentParser:
    """Build the six frozen campaign commands."""
    parser = argparse.ArgumentParser(prog="run_monitor_campaign")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--candidate-registry", type=Path, required=True)
    prepare.add_argument("--development-manifest", type=Path, required=True)
    prepare.add_argument("--dataset-lock", type=Path, required=True)
    prepare.add_argument("--staging", type=Path, required=True)
    for name in ("run", "resume", "status", "close"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--campaign", type=Path, required=True)
    materialize.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one frozen campaign operation."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_campaign(
            args.candidate_registry,
            args.development_manifest,
            args.dataset_lock,
            args.staging,
        )
        print(path)
        return 0
    if args.command == "status":
        print(canonical_json_bytes(load_campaign(args.campaign)).decode(), end="")
        return 0
    if args.command in {"run", "resume"}:
        _run_fitting(args.campaign, resume=args.command == "resume")
        return 0
    if args.command == "close":
        _close(args.campaign)
        return 0
    path = materialize_campaign(args.campaign, args.artifact_root)
    print(path)
    return 0


def _run_fitting(campaign: Path, *, resume: bool) -> None:
    """Run or resume exactly one candidate fitting process."""
    state = load_campaign(campaign)
    profile, candidate = current_attempt_identity(state)
    current = next(
        (
            item
            for item in state.attempts
            if (item.profile, item.candidate_name) == (profile, candidate)
        ),
        None,
    )
    if resume and (current is None or current.state not in {"fitting", "calibrating"}):
        raise CampaignError("resume needs the current incomplete candidate")
    if not resume and current is not None:
        raise CampaignError("run cannot replace the current candidate")
    command = (
        sys.executable,
        str(REPO_ROOT / "scripts/train_monitor.py"),
        "--formal-campaign",
        str(campaign.resolve()),
    )
    execute_fitting_process(
        campaign,
        external_time=github_server_time(),
        repo_root=REPO_ROOT,
        command=command,
    )


def _close(campaign: Path) -> None:
    """Publish and record one externally timed campaign marker."""
    state = load_campaign(campaign)
    observed = github_server_time()
    reason = (
        "terminal_completion"
        if len(state.completed_profiles) == 5 and observed <= state.candidate_cutoff
        else "cutoff_elapsed"
    )
    tag, assets = prepare_campaign_close(
        campaign,
        reason=reason,
        external_time=observed,
    )
    release = publish_marker_release(
        GitHubReleaseTransport(),
        tag=tag,
        target_revision=state.training_revision,
        assets=assets,
    )
    if release.published_at is None:
        raise CampaignError("the campaign marker has no publication time")
    close_campaign(
        campaign,
        reason=reason,
        published_at=release.published_at,
        release_id=release.release_id,
        release_api_url=release.api_url,
    )


def github_server_time() -> datetime:
    """Read the authoritative GitHub response time."""
    result = _gh(("api", "-i", "/rate_limit"))
    header = next(
        (
            line.partition(":")[2].strip()
            for line in result.stdout.splitlines()
            if line.lower().startswith("date:")
        ),
        None,
    )
    if header is None:
        raise CampaignError("the external time check has no Date header")
    parsed = parsedate_to_datetime(header)
    if parsed.tzinfo is None:
        raise CampaignError("the external time check is not timezone aware")
    return parsed.astimezone(UTC)


def _remote_release(value: dict[str, Any]) -> RemoteRelease:
    """Convert one GitHub release response."""
    published = value.get("published_at")
    return RemoteRelease(
        release_id=str(value["id"]),
        tag=str(value["tag_name"]),
        target_revision=str(value["target_commitish"]),
        api_url=str(value["url"]),
        draft=bool(value["draft"]),
        published_at=(
            datetime.fromisoformat(published.replace("Z", "+00:00"))
            if published
            else None
        ),
        assets=tuple(
            RemoteAsset(str(item["name"]), str(item["browser_download_url"]))
            for item in value["assets"]
        ),
    )


def _gh_json(arguments: tuple[str, ...]) -> dict[str, Any]:
    """Run one GitHub command and parse its JSON object."""
    return json.loads(_gh(arguments).stdout)


def _gh(
    arguments: tuple[str, ...],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run one authenticated GitHub CLI operation."""
    return subprocess.run(
        ("gh", *arguments),
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=text,
    )


if __name__ == "__main__":
    raise SystemExit(main())
