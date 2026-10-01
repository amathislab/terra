from terra._revision import GIT_COMMIT_FILE, write_git_commit


def test_write_git_commit_uses_explicit_environment(monkeypatch, tmp_path):
    commit = "a" * 40
    monkeypatch.setenv("TERRA_GIT_COMMIT", commit)
    marker = write_git_commit(tmp_path / "result")
    assert marker == (tmp_path / "result" / GIT_COMMIT_FILE).resolve()
    assert marker.read_text() == f"{commit}\n"


def test_source_copy_records_unknown(monkeypatch, tmp_path):
    monkeypatch.delenv("TERRA_GIT_COMMIT", raising=False)
    marker = write_git_commit(tmp_path / "result", repo_root=tmp_path)
    assert marker.read_text() == "unknown\n"
