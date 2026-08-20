"""Test ACL 3-level permissions."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hakua_memory.composite import CompositeMemory
from hakua_memory.rag.models import AclEntry, AclCheckResult


def test_acl_three_levels() -> None:
    """Test read/write/delete ACL separation."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test"
        cm = CompositeMemory(root)

        # Ingest a document
        result = cm.ingest_markdown(
            "# Test\nContent",
            title="ACL Test",
        )
        doc_id = result["document_id"]

        # Grant read to alice
        cm.grant_access(doc_id, "alice", "read")
        alice_perms = cm.check_access(doc_id, "alice")
        assert alice_perms["can_read"] is True
        assert alice_perms["can_write"] is False
        assert alice_perms["can_delete"] is False

        # Grant write to bob
        cm.grant_access(doc_id, "bob", "write")
        bob_perms = cm.check_access(doc_id, "bob")
        assert bob_perms["can_read"] is False
        assert bob_perms["can_write"] is True
        assert bob_perms["can_delete"] is False

        # Grant delete to carol
        cm.grant_access(doc_id, "carol", "delete")
        carol_perms = cm.check_access(doc_id, "carol")
        assert carol_perms["can_read"] is False
        assert carol_perms["can_write"] is False
        assert carol_perms["can_delete"] is True

        print("✅ ACL 3-level separation: PASSED")
    finally:
        cm.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_acl_revoke() -> None:
    """Test ACL revocation."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test"
        cm = CompositeMemory(root)

        result = cm.ingest_markdown("# Test\nContent", title="Revoke Test")
        doc_id = result["document_id"]

        # Grant and then revoke
        cm.grant_access(doc_id, "alice", "read")
        perms = cm.check_access(doc_id, "alice")
        assert perms["can_read"] is True

        cm.revoke_access(doc_id, "alice", "read")
        perms = cm.check_access(doc_id, "alice")
        assert perms["can_read"] is False

        print("✅ ACL revocation: PASSED")
    finally:
        cm.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_acl_validation() -> None:
    """Test ACL permission validation."""
    # Valid permissions
    AclEntry(document_id="d1", principal="alice", permission="read")
    AclEntry(document_id="d1", principal="bob", permission="write")
    AclEntry(document_id="d1", principal="carol", permission="delete")

    # Invalid permission
    try:
        AclEntry(document_id="d1", principal="alice", permission="admin")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid ACL permission" in str(e)

    # Empty document_id
    try:
        AclEntry(document_id="", principal="alice", permission="read")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "document_id" in str(e)

    print("✅ ACL validation: PASSED")


def test_acl_check_result() -> None:
    """Test AclCheckResult helper methods."""
    result = AclCheckResult(
        document_id="d1",
        principal="alice",
        can_read=True,
        can_write=True,
        can_delete=False,
        permissions=["read", "write"],
    )

    assert result.has("read") is True
    assert result.has("write") is True
    assert result.has("delete") is False
    assert result.has_all("read", "write") is True
    assert result.has_all("read", "delete") is False
    assert result.has_any("read", "delete") is True
    assert result.has_any("delete") is False

    print("✅ AclCheckResult helpers: PASSED")


def test_acl_department() -> None:
    """Test department-scoped ACL."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test"
        cm = CompositeMemory(root)

        result = cm.ingest_markdown("# Test\nContent", title="Dept Test")
        doc_id = result["document_id"]

        # Grant department-level access
        cm.grant_access(doc_id, "alice", "read", department="Engineering")
        dept_perms = cm.check_access_department(doc_id, "Engineering")
        assert "read" in dept_perms

        print("✅ Department ACL: PASSED")
    finally:
        cm.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_acl_validation()
    test_acl_check_result()
    test_acl_three_levels()
    test_acl_revoke()
    test_acl_department()
    print("\n★ ALL ACL TESTS PASSED ★")
