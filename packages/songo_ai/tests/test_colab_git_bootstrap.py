import subprocess
from pathlib import Path

import pytest

from songo_ai.training.colab_git_bootstrap import GitBootstrapError, bootstrap_repository, run_visible


RUNNER = "apps/trainer/scripts/run_srn_lot46.py"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


@pytest.fixture
def remote(tmp_path):
    source=tmp_path/"source"; source.mkdir(); git("init","-b","dev",cwd=source)
    git("config","user.email","test@example.invalid",cwd=source);git("config","user.name","Test",cwd=source)
    path=source/RUNNER;path.parent.mkdir(parents=True);path.write_text("print('lot46')\n")
    git("add",".",cwd=source);git("commit","-m","initial",cwd=source)
    bare=tmp_path/"remote.git";git("clone","--bare",str(source),str(bare))
    return bare,git("rev-parse","HEAD",cwd=source)


def boot(remote, destination, **kwargs):
    bare,commit=remote
    return bootstrap_repository(url=str(bare),branch="dev",preferred=destination,runner_path=RUNNER,**kwargs),commit


def test_first_clone_and_clean_rerun(remote,tmp_path):
    first,commit=boot(remote,tmp_path/"repo"); second,_=boot(remote,tmp_path/"repo")
    assert first.commit==commit and not first.reused
    assert second.repository==first.repository and second.reused


def test_dirty_repository_is_preserved_and_new_clone_is_used(remote,tmp_path):
    first,_=boot(remote,tmp_path/"repo"); tracked=first.repository/RUNNER;tracked.write_text("local science\n")
    second,_=boot(remote,tmp_path/"repo",clock=lambda:123)
    assert second.repository!=first.repository and second.dirty_repository_preserved==first.repository
    assert tracked.read_text()=="local science\n"


def test_untracked_files_are_preserved(remote,tmp_path):
    first,_=boot(remote,tmp_path/"repo"); artifact=first.repository/"checkpoint.pt";artifact.write_bytes(b"science")
    second,_=boot(remote,tmp_path/"repo",clock=lambda:456)
    assert second.repository!=first.repository and artifact.read_bytes()==b"science"


def test_missing_remote_branch(remote,tmp_path):
    with pytest.raises(GitBootstrapError,match="ls-remote"):
        bootstrap_repository(url=str(remote[0]),branch="absent",preferred=tmp_path/"repo",runner_path=RUNNER)


@pytest.mark.parametrize("failed_verb",["fetch","checkout"])
def test_fetch_and_checkout_errors_include_stderr(remote,tmp_path,failed_verb):
    real=subprocess.run
    def failing(command,**kwargs):
        if failed_verb in command:
            return subprocess.CompletedProcess(command,9,"partial stdout","diagnostic stderr")
        return real(command,**kwargs)
    with pytest.raises(GitBootstrapError,match="diagnostic stderr"):
        bootstrap_repository(url=str(remote[0]),branch="dev",preferred=tmp_path/"repo",runner_path=RUNNER,runner=failing)


def test_pinned_commit_does_not_follow_new_remote_commit(remote,tmp_path):
    first,commit=boot(remote,tmp_path/"repo")
    work=tmp_path/"work";git("clone",str(remote[0]),str(work));git("config","user.email","x@y",cwd=work);git("config","user.name","X",cwd=work)
    (work/"new.txt").write_text("new");git("add",".",cwd=work);git("commit","-m","new",cwd=work);git("push","origin","dev",cwd=work)
    pinned=bootstrap_repository(url=str(remote[0]),branch="dev",preferred=first.repository,runner_path=RUNNER,pinned_commit=commit)
    assert pinned.commit==commit


def test_run_visible_reports_command_failure(capsys):
    def fail(command,**kwargs): return subprocess.CompletedProcess(command,7,"out","err")
    with pytest.raises(GitBootstrapError):run_visible(["git","fetch"],runner=fail)
    shown=capsys.readouterr().out
    assert "exit=7" in shown and "stdout:\nout" in shown and "stderr:\nerr" in shown
