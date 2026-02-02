#!/usr/bin/env python3
"""
Itential GAIT MCP - Git for AI Tracking for FlowAI

Runs directly on the IAG server with per-agent local storage.
Each agent gets its own subfolder for isolated conversation tracking.

Usage:
    pip install -r requirements.txt
    python gait_mcp.py

FlowAI passes agent_id in each tool call to identify the agent/session.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------
# Logging (stderr only)
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("ItentialGaitMCP")

# ---------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        log.error("FastMCP not found. Install with: pip install mcp")
        raise

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
GAIT_STORAGE_ROOT = Path(os.environ.get("GAIT_STORAGE_ROOT", "/opt/gait/agents"))

mcp = FastMCP("GAIT")

# ---------------------------------------------------------------------
# Simple Local GAIT Implementation
# ---------------------------------------------------------------------
@dataclass
class MemoryItem:
    turn_id: str
    commit_id: str
    note: str

@dataclass
class MemoryManifest:
    items: List[MemoryItem]


class LocalGaitRepo:
    """
    Simple GAIT repo implementation with local filesystem storage.
    Each agent_id gets its own isolated storage folder.
    """

    def __init__(self, agent_id: str):
        # Sanitize agent_id
        safe_id = "".join(c for c in agent_id if c.isalnum() or c in "-_.")
        if not safe_id:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        self.agent_id = safe_id
        self._root = GAIT_STORAGE_ROOT / safe_id
        self._gait_dir = self._root / ".gait"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def gait_dir(self) -> Path:
        return self._gait_dir

    def _objects_dir(self) -> Path:
        return self._gait_dir / "objects"

    def _refs_dir(self) -> Path:
        return self._gait_dir / "refs" / "heads"

    def _memory_refs_dir(self) -> Path:
        return self._gait_dir / "refs" / "memory"

    def _hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _write_object(self, data: dict) -> str:
        content = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        oid = self._hash(content)
        obj_path = self._objects_dir() / oid[:2] / oid[2:]
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        obj_path.write_bytes(content)
        return oid

    def _read_object(self, oid: str) -> dict:
        obj_path = self._objects_dir() / oid[:2] / oid[2:]
        if not obj_path.exists():
            raise FileNotFoundError(f"Object not found: {oid}")
        return json.loads(obj_path.read_bytes())

    def init(self) -> dict:
        """Initialize the GAIT repo structure."""
        existed = self._gait_dir.exists()

        if not existed:
            self._gait_dir.mkdir(parents=True, exist_ok=True)
            self._objects_dir().mkdir(parents=True, exist_ok=True)
            self._refs_dir().mkdir(parents=True, exist_ok=True)
            self._memory_refs_dir().mkdir(parents=True, exist_ok=True)

            # Initialize HEAD
            (self._gait_dir / "HEAD").write_text("ref: refs/heads/main\n")

            # Initialize empty memory
            manifest = {"version": 1, "items": []}
            mem_id = self._write_object(manifest)
            self._write_memory_ref("main", mem_id)

            log.info(f"Initialized GAIT repo for agent: {self.agent_id}")

        return {
            "existed": existed,
            "root": str(self._root),
            "gait_dir": str(self._gait_dir)
        }

    def current_branch(self) -> str:
        head_path = self._gait_dir / "HEAD"
        if head_path.exists():
            content = head_path.read_text().strip()
            if content.startswith("ref: refs/heads/"):
                return content.replace("ref: refs/heads/", "")
        return "main"

    def _read_ref(self, name: str) -> Optional[str]:
        ref_path = self._refs_dir() / name
        if ref_path.exists():
            val = ref_path.read_text().strip()
            return val if val else None
        return None

    def _write_ref(self, name: str, commit_id: str) -> None:
        ref_path = self._refs_dir() / name
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(commit_id)

    def _read_memory_ref(self, branch: str) -> str:
        mem_path = self._memory_refs_dir() / branch
        if mem_path.exists():
            return mem_path.read_text().strip()
        return ""

    def _write_memory_ref(self, branch: str, memory_id: str) -> None:
        mem_path = self._memory_refs_dir() / branch
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text(memory_id)

    def head_commit_id(self) -> Optional[str]:
        return self._read_ref(self.current_branch())

    def get_commit(self, cid: str) -> dict:
        commit = self._read_object(cid)
        commit["_id"] = cid
        return commit

    def get_turn(self, tid: str) -> dict:
        return self._read_object(tid)

    def record_turn(self, turn_data: dict, message: str = "") -> Tuple[str, str]:
        """Record a turn and create a commit."""
        turn_id = self._write_object(turn_data)

        branch = self.current_branch()
        parent = self._read_ref(branch)

        commit = {
            "type": "commit",
            "kind": "turn",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "turn_ids": [turn_id],
            "parents": [parent] if parent else []
        }
        commit_id = self._write_object(commit)
        self._write_ref(branch, commit_id)

        return (turn_id, commit_id)

    def get_memory(self) -> MemoryManifest:
        branch = self.current_branch()
        mem_id = self._read_memory_ref(branch)

        if not mem_id:
            return MemoryManifest(items=[])

        try:
            manifest = self._read_object(mem_id)
            items = [
                MemoryItem(
                    turn_id=it.get("turn_id", ""),
                    commit_id=it.get("commit_id", ""),
                    note=it.get("note", "")
                )
                for it in manifest.get("items", [])
            ]
            return MemoryManifest(items=items)
        except FileNotFoundError:
            return MemoryManifest(items=[])

    def pin_commit(self, commit: Optional[str] = None, last: bool = True, note: str = "") -> str:
        branch = self.current_branch()

        if commit:
            commit_id = commit
        elif last:
            commit_id = self.head_commit_id()
            if not commit_id:
                raise ValueError("No commits to pin")
        else:
            raise ValueError("No commit specified")

        # Get turn ID from commit
        commit_data = self.get_commit(commit_id)
        turn_ids = commit_data.get("turn_ids", [])
        turn_id = turn_ids[0] if turn_ids else ""

        # Get current manifest
        mem_id = self._read_memory_ref(branch)
        if mem_id:
            try:
                manifest = self._read_object(mem_id)
            except FileNotFoundError:
                manifest = {"version": 1, "items": []}
        else:
            manifest = {"version": 1, "items": []}

        # Add item
        manifest["items"].append({
            "turn_id": turn_id,
            "commit_id": commit_id,
            "note": note
        })

        new_mem_id = self._write_object(manifest)
        self._write_memory_ref(branch, new_mem_id)
        return new_mem_id

    def unpin_index(self, index: int) -> str:
        branch = self.current_branch()
        mem_id = self._read_memory_ref(branch)

        if not mem_id:
            raise ValueError("No memory manifest")

        manifest = self._read_object(mem_id)
        items = manifest.get("items", [])

        if index < 1 or index > len(items):
            raise ValueError(f"Invalid index: {index}")

        items.pop(index - 1)
        manifest["items"] = items

        new_mem_id = self._write_object(manifest)
        self._write_memory_ref(branch, new_mem_id)
        return new_mem_id

    def build_context_bundle(self, full: bool = False) -> dict:
        manifest = self.get_memory()
        items = []

        for it in manifest.items:
            item_data = {
                "turn_id": it.turn_id,
                "commit_id": it.commit_id,
                "note": it.note
            }
            if full and it.turn_id:
                try:
                    item_data["turn"] = self.get_turn(it.turn_id)
                except FileNotFoundError:
                    pass
            items.append(item_data)

        return {"items": items}

    def reset_branch(self, commit_id: str) -> str:
        branch = self.current_branch()
        self._write_ref(branch, commit_id)
        return commit_id

    def reset_memory_to_commit(self, branch: str, commit_id: Optional[str]) -> str:
        manifest = {"version": 1, "items": []}
        new_mem_id = self._write_object(manifest)
        self._write_memory_ref(branch, new_mem_id)
        return new_mem_id

    def create_branch(self, name: str, from_commit: Optional[str] = None, inherit_memory: bool = True) -> None:
        ref_path = self._refs_dir() / name
        if ref_path.exists():
            raise FileExistsError(f"Branch '{name}' already exists")

        commit_id = from_commit if from_commit else (self.head_commit_id() or "")
        self._write_ref(name, commit_id)

        if inherit_memory:
            current_mem = self._read_memory_ref(self.current_branch())
            if current_mem:
                self._write_memory_ref(name, current_mem)

    def checkout(self, name: str) -> None:
        ref_path = self._refs_dir() / name
        if not ref_path.exists():
            raise FileNotFoundError(f"Branch '{name}' does not exist")
        (self._gait_dir / "HEAD").write_text(f"ref: refs/heads/{name}\n")

    def merge(self, source: str, message: str = "", with_memory: bool = False) -> str:
        source_head = self._read_ref(source)
        if not source_head:
            raise FileNotFoundError(f"Branch '{source}' not found")

        branch = self.current_branch()
        current_head = self._read_ref(branch)

        parents = []
        if current_head:
            parents.append(current_head)
        parents.append(source_head)

        merge_commit = {
            "type": "commit",
            "kind": "merge",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": message or f"Merge {source} into {branch}",
            "turn_ids": [],
            "parents": parents
        }
        merge_id = self._write_object(merge_commit)
        self._write_ref(branch, merge_id)

        return merge_id

    def summarize_and_squash(self, last: int = 10, mode: str = "soft", message: str = "", include_merges: bool = False) -> dict:
        branch = self.current_branch()
        head = self._read_ref(branch)

        if not head:
            raise ValueError("No commits to squash")

        # Walk back and collect
        commits_to_squash = []
        all_turn_ids = []
        current = head
        seen = set()

        while current and current not in seen and len(commits_to_squash) < last:
            seen.add(current)
            try:
                commit = self._read_object(current)
            except FileNotFoundError:
                break

            parents = commit.get("parents", [])
            if len(parents) > 1 and not include_merges:
                current = parents[0] if parents else ""
                continue

            commits_to_squash.append(current)
            all_turn_ids.extend(commit.get("turn_ids", []))
            current = parents[0] if parents else ""

        base_parent = current

        # Create summary turn
        summary_turn = {
            "type": "summary",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "squashed_turns": all_turn_ids,
            "user": {"text": f"[Squashed {len(commits_to_squash)} commits]"},
            "assistant": {"text": message or "Summary of squashed commits"}
        }
        summary_turn_id = self._write_object(summary_turn)

        # Create squash commit
        squash_commit = {
            "type": "commit",
            "kind": "squash",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": message or f"Squash {len(commits_to_squash)} commits",
            "turn_ids": [summary_turn_id],
            "parents": [base_parent] if base_parent else []
        }
        new_head = self._write_object(squash_commit)
        self._write_ref(branch, new_head)

        return {
            "branch": branch,
            "mode": mode,
            "old_head": head,
            "new_head": new_head,
            "squashed_commits": commits_to_squash,
            "base_parent": base_parent,
            "turns_squashed": len(all_turn_ids),
            "summary_turn_id": summary_turn_id
        }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def short_oid(oid: str) -> str:
    return oid[:8] if oid else ""


def _err(msg: str, *, detail: str = "", **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "error": msg}
    if detail:
        out["detail"] = detail
    out.update(extra)
    return out


def _get_repo(agent_id: str) -> Tuple[Optional[LocalGaitRepo], Optional[Dict[str, Any]]]:
    """Get or create a repo for the given agent_id."""
    if not agent_id or not agent_id.strip():
        return (None, _err("agent_id is required"))

    try:
        return (LocalGaitRepo(agent_id), None)
    except Exception as e:
        return (None, _err("Failed to get repo", detail=str(e)))


def _resolve_commit_prefix(repo: LocalGaitRepo, head: str, prefix: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("empty commit prefix")

    cid = head
    seen = set()
    while cid and cid not in seen:
        seen.add(cid)
        if cid.startswith(prefix):
            return cid
        c = repo.get_commit(cid)
        parents = c.get("parents") or []
        cid = parents[0] if parents else ""
    raise ValueError(f"Unknown commit: {prefix}")


def _resolve_revert_target(repo: LocalGaitRepo, target: str) -> str:
    head = repo.head_commit_id() or ""
    if not head:
        raise ValueError("No HEAD commit")

    t = (target or "HEAD~1").strip()

    m = re.fullmatch(r"HEAD~(\d+)", t.upper())
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise ValueError("HEAD~N must be >= 1")

        cid = head
        for _ in range(n):
            c = repo.get_commit(cid)
            parents = c.get("parents") or []
            cid = parents[0] if parents else ""
            if not cid:
                break
        return cid

    return _resolve_commit_prefix(repo, head, t)


def mcp_tool(fn):
    """Decorator for MCP tools with error handling."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.exception("tool failed: %s", fn.__name__)
            return _err(f"{fn.__name__} failed", detail=str(e))
    wrapper.__signature__ = inspect.signature(fn)
    return wrapper


# ---------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------
@mcp_tool
@mcp.tool()
def gait_init(agent_id: str) -> Dict[str, Any]:
    """Initialize GAIT tracking for an agent.

    Args:
        agent_id: Unique identifier for this agent/session
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    result = repo.init()
    return {"ok": True, "agent_id": agent_id, **result}


@mcp_tool
@mcp.tool()
def gait_status(agent_id: str) -> Dict[str, Any]:
    """Show GAIT repo status for an agent.

    Args:
        agent_id: Unique identifier for this agent/session
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    if not repo.gait_dir.exists():
        return _err("GAIT not initialized. Call gait_init first.", agent_id=agent_id)

    return {
        "ok": True,
        "agent_id": agent_id,
        "root": str(repo.root),
        "branch": repo.current_branch(),
        "head": repo.head_commit_id() or ""
    }


@mcp_tool
@mcp.tool()
def gait_record_turn(
    agent_id: str,
    user_text: str = "",
    assistant_text: str = "",
    artifacts: Optional[List[Dict[str, str]]] = None,
    note: str = "flowai"
) -> Dict[str, Any]:
    """Record a conversation turn with optional code artifacts.

    Args:
        agent_id: Unique identifier for this agent/session
        user_text: The user's input text
        assistant_text: The assistant's response text
        artifacts: List of {"path": "...", "content": "..."} for code files
        note: Optional note/tag for this turn
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    if not repo.gait_dir.exists():
        repo.init()

    # Build turn data
    turn_data = {
        "version": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user": {"text": user_text},
        "assistant": {"text": assistant_text},
        "context": {"artifacts": artifacts or []},
        "model": {"provider": "flowai"}
    }

    turn_id, commit_id = repo.record_turn(turn_data, message=note)

    return {
        "ok": True,
        "agent_id": agent_id,
        "commit": short_oid(commit_id),
        "turn_id": short_oid(turn_id),
        "artifacts_tracked": len(artifacts or [])
    }


@mcp_tool
@mcp.tool()
def gait_log(agent_id: str, limit: int = 20) -> Dict[str, Any]:
    """List recent commits for an agent.

    Args:
        agent_id: Unique identifier for this agent/session
        limit: Maximum number of commits to return
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    if not repo.gait_dir.exists():
        return _err("GAIT not initialized", agent_id=agent_id)

    commits = []
    head = repo.head_commit_id()
    if not head:
        return {"ok": True, "agent_id": agent_id, "branch": repo.current_branch(), "commits": []}

    cid = head
    seen = set()
    while cid and cid not in seen and len(commits) < limit:
        seen.add(cid)
        try:
            c = repo.get_commit(cid)
        except FileNotFoundError:
            break

        parents = c.get("parents") or []
        commits.append({
            "commit": short_oid(cid),
            "id": cid,
            "created_at": c.get("created_at", ""),
            "kind": c.get("kind", ""),
            "message": c.get("message", ""),
            "turns": len(c.get("turn_ids", []))
        })
        cid = parents[0] if parents else ""

    return {"ok": True, "agent_id": agent_id, "branch": repo.current_branch(), "commits": commits}


@mcp_tool
@mcp.tool()
def gait_show(agent_id: str, commit: str = "HEAD") -> Dict[str, Any]:
    """Show a commit with its turns and artifacts.

    Args:
        agent_id: Unique identifier for this agent/session
        commit: Commit ID or 'HEAD'
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    head = repo.head_commit_id()
    if not head:
        return _err("No commits yet", agent_id=agent_id)

    cid = head if commit in ("", "HEAD", "@") else _resolve_commit_prefix(repo, head, commit)
    c = repo.get_commit(cid)

    turns = []
    for tid in c.get("turn_ids", []):
        t = repo.get_turn(tid)
        turns.append({
            "turn_id": short_oid(tid),
            "user": (t.get("user") or {}).get("text", ""),
            "assistant": (t.get("assistant") or {}).get("text", ""),
            "artifacts": t.get("context", {}).get("artifacts", [])
        })

    return {
        "ok": True,
        "agent_id": agent_id,
        "commit": short_oid(cid),
        "created_at": c.get("created_at", ""),
        "message": c.get("message", ""),
        "turns": turns
    }


@mcp_tool
@mcp.tool()
def gait_resume(agent_id: str, turns: int = 10) -> Dict[str, Any]:
    """Restore conversation context from history.

    Args:
        agent_id: Unique identifier for this agent/session
        turns: Number of recent turns to restore
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    head = repo.head_commit_id()
    if not head:
        return _err("No commits yet", agent_id=agent_id)

    collected = []
    cid = head
    seen = set()

    while cid and cid not in seen and len(collected) < turns:
        seen.add(cid)
        c = repo.get_commit(cid)

        for tid in c.get("turn_ids", []):
            if len(collected) >= turns:
                break
            t = repo.get_turn(tid)
            user_txt = (t.get("user") or {}).get("text", "")
            asst_txt = (t.get("assistant") or {}).get("text", "")
            artifacts = t.get("context", {}).get("artifacts", [])

            if user_txt or asst_txt:
                collected.append({
                    "user": user_txt,
                    "assistant": asst_txt,
                    "artifacts": artifacts
                })

        parents = c.get("parents") or []
        cid = parents[0] if parents else ""

    collected.reverse()

    return {
        "ok": True,
        "agent_id": agent_id,
        "branch": repo.current_branch(),
        "head": short_oid(head),
        "turns_restored": len(collected),
        "history": collected
    }


@mcp_tool
@mcp.tool()
def gait_memory(agent_id: str) -> Dict[str, Any]:
    """List pinned memory items.

    Args:
        agent_id: Unique identifier for this agent/session
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    manifest = repo.get_memory()
    items = [
        {"index": i, "turn": short_oid(it.turn_id), "commit": short_oid(it.commit_id), "note": it.note}
        for i, it in enumerate(manifest.items, start=1)
    ]

    return {"ok": True, "agent_id": agent_id, "branch": repo.current_branch(), "items": items}


@mcp_tool
@mcp.tool()
def gait_pin(agent_id: str, commit: Optional[str] = None, note: str = "") -> Dict[str, Any]:
    """Pin a commit to memory.

    Args:
        agent_id: Unique identifier for this agent/session
        commit: Commit ID to pin (default: last commit)
        note: Optional note for this pin
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    mem_id = repo.pin_commit(commit, last=True, note=note)
    return {"ok": True, "agent_id": agent_id, "memory_id": short_oid(mem_id)}


@mcp_tool
@mcp.tool()
def gait_unpin(agent_id: str, index: int) -> Dict[str, Any]:
    """Unpin a memory item by index.

    Args:
        agent_id: Unique identifier for this agent/session
        index: 1-based index from gait_memory
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    mem_id = repo.unpin_index(index)
    return {"ok": True, "agent_id": agent_id, "unpinned": index, "memory_id": short_oid(mem_id)}


@mcp_tool
@mcp.tool()
def gait_context(agent_id: str, full: bool = False) -> Dict[str, Any]:
    """Build context bundle from pinned memory.

    Args:
        agent_id: Unique identifier for this agent/session
        full: Include full turn data
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    bundle = repo.build_context_bundle(full=full)
    return {"ok": True, "agent_id": agent_id, "bundle": bundle}


@mcp_tool
@mcp.tool()
def gait_revert(agent_id: str, target: str = "HEAD~1") -> Dict[str, Any]:
    """Rewind history to a previous commit.

    Args:
        agent_id: Unique identifier for this agent/session
        target: Target commit (e.g., HEAD~1, HEAD~3, or commit prefix)
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    head_before = repo.head_commit_id()
    if not head_before:
        return _err("No commits to revert", agent_id=agent_id)

    new_head = _resolve_revert_target(repo, target)
    repo.reset_branch(new_head)
    repo.reset_memory_to_commit(repo.current_branch(), new_head)

    return {
        "ok": True,
        "agent_id": agent_id,
        "head_before": short_oid(head_before),
        "head_after": short_oid(new_head) if new_head else "(empty)",
        "instruction": "Call gait_resume to restore context"
    }


@mcp_tool
@mcp.tool()
def gait_branch(agent_id: str, name: str, from_commit: Optional[str] = None) -> Dict[str, Any]:
    """Create a new branch.

    Args:
        agent_id: Unique identifier for this agent/session
        name: Branch name
        from_commit: Optional commit to branch from
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    try:
        repo.create_branch(name, from_commit=from_commit)
        return {"ok": True, "agent_id": agent_id, "created": name}
    except FileExistsError:
        return {"ok": True, "agent_id": agent_id, "branch": name, "note": "already exists"}


@mcp_tool
@mcp.tool()
def gait_checkout(agent_id: str, name: str) -> Dict[str, Any]:
    """Switch to a branch.

    Args:
        agent_id: Unique identifier for this agent/session
        name: Branch name
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    repo.checkout(name)
    return {"ok": True, "agent_id": agent_id, "branch": name, "head": repo.head_commit_id() or ""}


@mcp_tool
@mcp.tool()
def gait_merge(agent_id: str, source: str, message: str = "") -> Dict[str, Any]:
    """Merge a branch into current branch.

    Args:
        agent_id: Unique identifier for this agent/session
        source: Branch to merge from
        message: Optional merge message
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    merge_id = repo.merge(source, message=message)
    return {
        "ok": True,
        "agent_id": agent_id,
        "merged": source,
        "into": repo.current_branch(),
        "commit": short_oid(merge_id)
    }


@mcp_tool
@mcp.tool()
def gait_squash(agent_id: str, last: int = 10, message: str = "") -> Dict[str, Any]:
    """Squash recent commits into one.

    Args:
        agent_id: Unique identifier for this agent/session
        last: Number of commits to squash
        message: Optional squash message
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err

    result = repo.summarize_and_squash(last=last, message=message)
    return {
        "ok": True,
        "agent_id": agent_id,
        "old_head": short_oid(result["old_head"]),
        "new_head": short_oid(result["new_head"]),
        "squashed": len(result["squashed_commits"]),
        "instruction": "Call gait_resume to restore context"
    }


# ---------------------------------------------------------------------
# Patch tool schemas for FlowAI compatibility
# ---------------------------------------------------------------------
def _patch_tool_schemas():
    try:
        tools = mcp._tool_manager._tools
        for name, tool in tools.items():
            if hasattr(tool, 'parameters') and isinstance(tool.parameters, dict):
                if 'required' not in tool.parameters:
                    tool.parameters['required'] = []
    except Exception as e:
        log.warning(f"Could not patch tool schemas: {e}")

_patch_tool_schemas()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # Ensure storage root exists
    GAIT_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    host = os.environ.get("GAIT_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("GAIT_MCP_PORT", "8000"))

    log.info(f"Itential GAIT MCP starting on {host}:{port}")
    log.info(f"Storage root: {GAIT_STORAGE_ROOT}")

    uvicorn.run(mcp.sse_app(), host=host, port=port)
