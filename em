#!/usr/bin/env python3.6

import argparse
import configparser
import dbm
import os
import pprint
import re
import shelve
import shutil
import signal
import subprocess
import sys
import time

import daemon
import pygit2

def _die(msg):
    print(msg, file=sys.stderr)
    exit(1)

def _ensure_proj(cb):
    def docmd(*args, **kwargs):
        if not os.path.isfile('.em.db'):
            _die(f'error: \'{os.path.abspath(".")}\' is not a project directory')
        cb(*args, **kwargs)
    return docmd

def proj_create(args, config, extra_args):
    tmpl_repo = config['project']['template_repo']
    try:
        pygit2.clone_repository(tmpl_repo, args.dest)
    except ValueError:
        _die('error: project already exists')

    # re-init the repo
    shutil.rmtree(os.path.join(args.dest, '.git'), ignore_errors=True)
    repo = pygit2.init_repository(args.dest)

    shelve.open(os.path.join(args.dest, '.em')).close()
    os.mkdir(os.path.join(args.dest, 'experiments'))

def _cleanup(name, emdb, repo):
    exper_dir = f'experiments/{name}'
    if not name in emdb and not os.path.isdir(exper_dir):
        return
    shutil.rmtree(exper_dir, ignore_errors=True)
    try:
        worktree = repo.lookup_worktree(name)
        if worktree is not None:
            worktree.prune(True)
    except pygit2.GitError:
        pass
    try:
        br = repo.lookup_branch(name)
        if br is not None:
            br.delete()
    except pygit2.GitError:
        pass
    del emdb[name]

UNCH = {pygit2.GIT_STATUS_CURRENT, pygit2.GIT_STATUS_IGNORED}

def run(args, config, prog_args):
    repo = pygit2.Repository('.')

    with shelve.open('.em', writeback=True) as emdb:
        if args.name in emdb:
            newp = input(f'Experiment {args.name} already exists. Recreate? [yN] ')
            if newp.lower() != 'y':
                exit()
            _cleanup(args.name, emdb, repo)
            br = None

        try:
            br = repo.lookup_branch(args.name)
        except pygit2.GitError:
            br = None

    tracked_exts = set(config['experiment']['track_files'].split(','))
    has_src_changes = has_changes = False
    for path,status in repo.status().items():
        ext = os.path.splitext(os.path.basename(path))[1][1:]
        changed = status not in UNCH
        has_changes = has_changes or changed
        if ext in tracked_exts:
            has_src_changes = has_src_changes or changed

    saved_state = None
    base_commit = head_commit = repo.head.target # an Oid
    sig = repo.default_signature
    if br is not None:
        if br.is_checked_out():
            _die('error: cannot run experiment on checked out branch')
        if has_src_changes:
            _die('error: not updating existing branch with source changes')
        base_commit = br.target
        # libgit only creates worktree from head commit, so move to the branch
        saved_state = repo.stash(sig, include_untracked=True)
        repo.reset(base_commit, pygit2.GIT_RESET_HARD)
        br.delete()
        br = None

    exper_dir = os.path.join('experiments', args.name)
    repo.add_worktree(args.name, exper_dir)

    if has_src_changes:
        repo.index.add_all([f'*.{ext}' for ext in tracked_exts])
        snap_tree = repo.index.write_tree() # an Oid
        repo.create_commit(f'refs/heads/{args.name}', sig, sig,
                           f'setup experiment: {args.name}', snap_tree,
                           [base_commit])
        # update the workdir to match updated index
        workdir = pygit2.Repository(exper_dir)
        workdir.reset(workdir.head.target, pygit2.GIT_RESET_HARD)

    os.symlink(os.path.abspath('data'), os.path.join(exper_dir, 'data'),
              target_is_directory=True)

    if saved_state is not None:
        repo.reset(head_commit.id, pygit2.GIT_RESET_HARD)
        repo.stash_pop()

    run_cmd = [config['experiment']['prog']] + \
            config['experiment']['prog_args'] + \
            prog_args
    env = os.environ
    if args.gpu:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu

    def _run_job():
        try:
            job = subprocess.Popen(run_cmd, cwd=exper_dir, env=env,
                                   stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
            with shelve.open('.em', writeback=True) as emdb:
                emdb[args.name] = {
                    'created': time.time(),
                    'status': 'running',
                    'pid': job.pid,
                }
            job.wait()
            with shelve.open('.em', writeback=True) as emdb:
                status = 'completed' if job.returncode == 0 else 'interrupted'
                emdb[args.name]['status'] = status
        except KeyboardInterrupt:
            with shelve.open('.em', writeback=True) as emdb:
                emdb[args.name]['status'] = 'interrupted'
        finally:
            with shelve.open('.em', writeback=True) as emdb:
                del emdb[args.name]['pid']

    if args.bg:
        with daemon.DaemonContext(working_directory=os.path.abspath(os.curdir)):
            _run_job()
    else:
        _run_job()

def clean(args, config, extra_args):
    repo = pygit2.Repository('.')

    with shelve.open('.em', writeback=True) as emdb:
        _cleanup(args.name, emdb, repo)

def ls(args, config, extra_args):
    cols = shutil.get_terminal_size((80, 20)).columns
    with shelve.open('.em') as emdb:
        linewidth = -1
        line_names = []
        names = sorted(emdb.keys())
        if not names:
            exit()
        for name in sorted(emdb.keys()):
            if linewidth + len(name) >= cols:
                linewidth = -1
                sys.stdout.write('\n')
            sys.stdout.write(f'{name} ')
            linewidth += len(name) + 1
        sys.stdout.write('\n')
        sys.stdout.flush()

def show(args, config, extra_args):
    with shelve.open('.em') as emdb:
        if not args.name in emdb:
            _die(f'error: no experiment named \'{args.name}\'')
        pprint.pprint(emdb[args.name])

def _ps(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def ctl(args, config, extra_args):
    with shelve.open('.em') as emdb:
        if not args.name in emdb:
            _die(f'error: no experiment named \'{args.name}\'')
        pid = emdb[args.name].get('pid')
        if not pid or not _ps(pid):
            _die(f'error: experiment \'{args.name}\' is not running')

    cmd = args.cmd[0]
    if cmd == 'stop':
        os.kill(pid, signal.SIGINT)
    else:
        with open(f'experiments/{args.name}/run/ctl', 'w') as f_ctl:
            print(' '.join(args.cmd), file=f_ctl)

#===============================================================================

if __name__ != '__main__':
    exit()

parser = argparse.ArgumentParser(description='Manage projects and experiments.')
parser.add_argument('--config', '-c', help='path to config file')
subparsers = parser.add_subparsers()

parser_create = subparsers.add_parser('proj', help='create a new project')
parser_create.add_argument('dest', help='the project destination')
parser_create.set_defaults(_cmd=proj_create)

parser_run = subparsers.add_parser('run', help='run an experiment')
parser_run.add_argument('name', help='the name of the experiment')
parser_run.add_argument('--gpu', '-g',
                        help='comma separated id(s) of the gpu to use. none is all')
parser_run.add_argument('--bg', action='store_true',
                        help='run the experiment in the background')
parser_run.set_defaults(_cmd=_ensure_proj(run))

parser_clean = subparsers.add_parser('clean', help='clean up an experiment')
parser_clean.add_argument('name', help='the name of the experiment')
parser_clean.set_defaults(_cmd=_ensure_proj(clean))

parser_list = subparsers.add_parser('list', help='list experiments')
parser_list.set_defaults(_cmd=_ensure_proj(ls))

parser_show = subparsers.add_parser('show', help='show details about an experiment')
parser_show.add_argument('name', help='the name of the experiment')
parser_show.set_defaults(_cmd=_ensure_proj(show))

parser_ctl = subparsers.add_parser('ctl', help='control a running experiment')
parser_ctl.add_argument('name', help='the name of the experiment')
parser_ctl.add_argument('cmd', nargs='+', help='the control signal to send')
parser_ctl.set_defaults(_cmd=_ensure_proj(ctl))

if len(sys.argv) == 1:
    parser.print_usage()
    exit()

args, extra_args = parser.parse_known_args()

# eventually upgrade to configparser
config = {
    'project': {
        'template_repo': 'https://github.com/nhynes/pytorch-proj.git',
    },
    'experiment': {
        'track_files': 'py,sh,txt',
        'prog': sys.executable,
        'prog_args': ['main.py'],
    },
}

try:
    args._cmd(args, config, extra_args)
except KeyboardInterrupt:
    pass
