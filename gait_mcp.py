#!/usr/bin/env python3
"""
Itential GAIT MCP - Git for AI Tracking for FlowAI

Runs directly on the IAG server with per-agent local storage.
Each agent gets its own subfolder for isolated conversation tracking.
"""

from __future__ import annotations

import hashlib
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

mcp = FastMCP("gait")

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
    """Simple GAIT repo with local filesystem storage."""

    def __init__(self, agent_id: str):
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
        with open(obj_path, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        return oid

    def _read_object(self, oid: str) -> dict:
        obj_path = self._objects_dir() / oid[:2] / oid[2:]
        if not obj_path.exists():
            raise FileNotFoundError(f"Object not found: {oid}")
        return json.loads(obj_path.read_bytes())

    def init(self) -> dict:
        existed = self._gait_dir.exists()
        if not existed:
            self._gait_dir.mkdir(parents=True, exist_ok=True)
            self._objects_dir().mkdir(parents=True, exist_ok=True)
            self._refs_dir().mkdir(parents=True, exist_ok=True)
            self._memory_refs_dir().mkdir(parents=True, exist_ok=True)
            (self._gait_dir / "HEAD").write_text("ref: refs/heads/main\n")
            manifest = {"version": 1, "items": []}
            mem_id = self._write_object(manifest)
            self._write_memory_ref("main", mem_id)
        return {"existed": existed, "root": str(self._root), "gait_dir": str(self._gait_dir)}

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
        # Write with explicit flush to ensure data is persisted
        with open(ref_path, 'w') as f:
            f.write(commit_id)
            f.flush()
            os.fsync(f.fileno())

    def _read_memory_ref(self, branch: str) -> str:
        mem_path = self._memory_refs_dir() / branch
        if mem_path.exists():
            return mem_path.read_text().strip()
        return ""

    def _write_memory_ref(self, branch: str, memory_id: str) -> None:
        mem_path = self._memory_refs_dir() / branch
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mem_path, 'w') as f:
            f.write(memory_id)
            f.flush()
            os.fsync(f.fileno())

    def head_commit_id(self) -> Optional[str]:
        return self._read_ref(self.current_branch())

    def get_commit(self, cid: str) -> dict:
        commit = self._read_object(cid)
        commit["_id"] = cid
        return commit

    def get_turn(self, tid: str) -> dict:
        return self._read_object(tid)

    def record_turn(self, turn_data: dict, message: str = "") -> Tuple[str, str]:
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
                MemoryItem(turn_id=it.get("turn_id", ""), commit_id=it.get("commit_id", ""), note=it.get("note", ""))
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
        commit_data = self.get_commit(commit_id)
        turn_ids = commit_data.get("turn_ids", [])
        turn_id = turn_ids[0] if turn_ids else ""
        mem_id = self._read_memory_ref(branch)
        if mem_id:
            try:
                manifest = self._read_object(mem_id)
            except FileNotFoundError:
                manifest = {"version": 1, "items": []}
        else:
            manifest = {"version": 1, "items": []}
        manifest["items"].append({"turn_id": turn_id, "commit_id": commit_id, "note": note})
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
            item_data = {"turn_id": it.turn_id, "commit_id": it.commit_id, "note": it.note}
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def short_oid(oid: str) -> str:
    return oid[:8] if oid else ""


def _get_repo(agent_id: str):
    if not agent_id or not agent_id.strip():
        return None, {"ok": False, "error": "agent_id is required"}
    try:
        return LocalGaitRepo(agent_id), None
    except Exception as e:
        return None, {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------
# MCP Tools - Using simple dict returns for FlowAI compatibility
# ---------------------------------------------------------------------

@mcp.tool()
def init(agent_id: str) -> dict:
    """
    Initialize GAIT tracking for an agent session.

    Args:
        agent_id: Unique identifier for this agent or session (e.g., "flowai-session-123")

    Returns:
        Status of initialization with storage path
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    result = repo.init()
    return {"ok": True, "agent_id": agent_id, **result}


@mcp.tool()
def status(agent_id: str) -> dict:
    """
    Get GAIT repository status for an agent.

    Args:
        agent_id: Unique identifier for this agent or session

    Returns:
        Current branch, HEAD commit, and storage path
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    if not repo.gait_dir.exists():
        return {"ok": False, "error": "GAIT not initialized. Call init first.", "agent_id": agent_id}
    return {
        "ok": True,
        "agent_id": agent_id,
        "root": str(repo.root),
        "branch": repo.current_branch(),
        "head": repo.head_commit_id() or ""
    }


@mcp.tool()
def record_turn(
    agent_id: str,
    user_text: str = "",
    assistant_text: str = "",
    note: str = "flowai"
) -> dict:
    """
    Record a conversation turn between user and assistant.

    Args:
        agent_id: Unique identifier for this agent or session
        user_text: The user's message or query
        assistant_text: The assistant's response
        note: Optional tag or label for this turn (default: flowai)

    Returns:
        Commit ID for the recorded turn
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    if not repo.gait_dir.exists():
        repo.init()
    turn_data = {
        "version": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user": {"text": user_text},
        "assistant": {"text": assistant_text},
        "context": {"artifacts": []},
        "model": {"provider": "flowai"}
    }
    turn_id, commit_id = repo.record_turn(turn_data, message=note)
    # Verify the ref was written
    head_after = repo.head_commit_id()
    return {
        "ok": True,
        "agent_id": agent_id,
        "commit": commit_id,
        "commit_short": short_oid(commit_id),
        "turn_id": short_oid(turn_id),
        "head": head_after,
        "branch": repo.current_branch()
    }


@mcp.tool()
def log(agent_id: str, limit: int = 20) -> dict:
    """
    List recent commits for an agent.

    Args:
        agent_id: Unique identifier for this agent or session
        limit: Maximum number of commits to return (default: 20)

    Returns:
        List of recent commits with metadata
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    if not repo.gait_dir.exists():
        return {"ok": True, "agent_id": agent_id, "message": "No commits yet - GAIT not initialized", "commits": []}
    commits = []
    head = repo.head_commit_id()
    if not head:
        return {"ok": True, "agent_id": agent_id, "branch": repo.current_branch(), "message": "No commits yet", "commits": []}
    cid = head
    seen = set()
    while cid and cid not in seen and len(commits) < limit:
        seen.add(cid)
        try:
            c = repo.get_commit(cid)
        except FileNotFoundError:
            # Object missing, stop walking but don't fail
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
    if not commits:
        return {"ok": True, "agent_id": agent_id, "branch": repo.current_branch(), "message": "No commits yet", "commits": []}
    return {"ok": True, "agent_id": agent_id, "branch": repo.current_branch(), "commits": commits}


@mcp.tool()
def show(agent_id: str, commit: str = "HEAD") -> dict:
    """
    Show details of a specific commit including conversation turns.

    Args:
        agent_id: Unique identifier for this agent or session
        commit: Commit ID or 'HEAD' for latest (default: HEAD)

    Returns:
        Commit details with user/assistant conversation
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    if not repo.gait_dir.exists():
        return {"ok": True, "agent_id": agent_id, "message": "No commits yet - GAIT not initialized", "turns": []}
    head = repo.head_commit_id()
    if not head:
        return {"ok": True, "agent_id": agent_id, "message": "No commits yet", "turns": []}
    cid = head if commit in ("", "HEAD", "@") else commit
    try:
        c = repo.get_commit(cid)
    except FileNotFoundError:
        return {"ok": True, "agent_id": agent_id, "message": f"Commit not found: {short_oid(cid)}", "turns": []}
    turns = []
    for tid in c.get("turn_ids", []):
        try:
            t = repo.get_turn(tid)
            turns.append({
                "turn_id": short_oid(tid),
                "user": (t.get("user") or {}).get("text", ""),
                "assistant": (t.get("assistant") or {}).get("text", "")
            })
        except FileNotFoundError:
            # Turn object missing, skip it
            turns.append({"turn_id": short_oid(tid), "error": "Turn data not found"})
    return {
        "ok": True,
        "agent_id": agent_id,
        "commit": short_oid(cid),
        "created_at": c.get("created_at", ""),
        "message": c.get("message", ""),
        "turns": turns
    }


@mcp.tool()
def resume(agent_id: str, turns: int = 10) -> dict:
    """
    Restore conversation context from history.

    Args:
        agent_id: Unique identifier for this agent or session
        turns: Number of recent turns to restore (default: 10)

    Returns:
        Conversation history for context restoration
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    if not repo.gait_dir.exists():
        return {"ok": True, "agent_id": agent_id, "message": "No history yet - GAIT not initialized", "history": []}
    head = repo.head_commit_id()
    if not head:
        return {"ok": True, "agent_id": agent_id, "message": "No history yet", "history": []}
    collected = []
    cid = head
    seen = set()
    while cid and cid not in seen and len(collected) < turns:
        seen.add(cid)
        try:
            c = repo.get_commit(cid)
        except FileNotFoundError:
            break
        for tid in c.get("turn_ids", []):
            if len(collected) >= turns:
                break
            try:
                t = repo.get_turn(tid)
                user_txt = (t.get("user") or {}).get("text", "")
                asst_txt = (t.get("assistant") or {}).get("text", "")
                if user_txt or asst_txt:
                    collected.append({"user": user_txt, "assistant": asst_txt})
            except FileNotFoundError:
                continue
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


@mcp.tool()
def memory(agent_id: str) -> dict:
    """
    List pinned memory items for an agent.

    Args:
        agent_id: Unique identifier for this agent or session

    Returns:
        List of pinned conversation items
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    if not repo.gait_dir.exists():
        return {"ok": True, "agent_id": agent_id, "message": "No memory yet - GAIT not initialized", "items": []}
    try:
        manifest = repo.get_memory()
        items = [
            {"index": i, "turn": short_oid(it.turn_id), "commit": short_oid(it.commit_id), "note": it.note}
            for i, it in enumerate(manifest.items, start=1)
        ]
        return {"ok": True, "agent_id": agent_id, "branch": repo.current_branch(), "items": items}
    except Exception as e:
        return {"ok": True, "agent_id": agent_id, "message": f"Could not read memory: {str(e)}", "items": []}


@mcp.tool()
def pin(agent_id: str, note: str = "") -> dict:
    """
    Pin the last commit to memory for future reference.

    Args:
        agent_id: Unique identifier for this agent or session
        note: Optional note describing why this is pinned

    Returns:
        Memory ID of the pinned item
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    try:
        mem_id = repo.pin_commit(note=note)
        return {"ok": True, "agent_id": agent_id, "memory_id": short_oid(mem_id)}
    except Exception as e:
        return {"ok": False, "error": str(e), "agent_id": agent_id}


@mcp.tool()
def unpin(agent_id: str, index: int) -> dict:
    """
    Remove a pinned memory item by its index.

    Args:
        agent_id: Unique identifier for this agent or session
        index: 1-based index from the memory list

    Returns:
        Updated memory ID
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    try:
        mem_id = repo.unpin_index(index)
        return {"ok": True, "agent_id": agent_id, "unpinned": index, "memory_id": short_oid(mem_id)}
    except Exception as e:
        return {"ok": False, "error": str(e), "agent_id": agent_id}


@mcp.tool()
def context(agent_id: str, full: bool = False) -> dict:
    """
    Build context bundle from pinned memory items.

    Args:
        agent_id: Unique identifier for this agent or session
        full: Include full turn data (default: False)

    Returns:
        Context bundle for AI consumption
    """
    repo, err = _get_repo(agent_id)
    if err:
        return err
    bundle = repo.build_context_bundle(full=full)
    return {"ok": True, "agent_id": agent_id, "bundle": bundle}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    GAIT_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    mcp.run()
