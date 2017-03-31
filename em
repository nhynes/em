#!/usr/bin/env python3.6

import argparse
import datetime
import os
import shelve
import shutil
import sys

import pygit2


def _die(msg, status=1):
    print(msg, file=sys.stderr)
    return status


def _ensure_proj(cb):
    def docmd(*args, **kwargs):
        if not os.path.isfile('.em.db'):
            curdir = os.path.abspath('.')
            return _die(f'error: "{curdir}" is not a project directory')
        cb(*args, **kwargs)
    return docmd


def _expath(*args):
    return os.path.abspath(os.path.join('experiments', *args))


def proj_create(args, config, extra_args):
    tmpl_repo = config['project']['template_repo']
    try:
        pygit2.clone_repository(tmpl_repo, args.dest)
    except ValueError:
        return _die('error: project already exists')

    # re-init the repo
    shutil.rmtree(os.path.join(args.dest, '.git'), ignore_errors=True)
    repo = pygit2.init_repository(args.dest)

    shelve.open(os.path.join(args.dest, '.em')).close()
    for d in ['experiments', 'data']:
        os.mkdir(os.path.join(args.dest, d))


def _cleanup(name, emdb, repo):
    exper_dir = _expath(name)
    if name not in emdb and not os.path.isdir(exper_dir):
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


def _tstamp():
    import time
    return datetime.datetime.fromtimestamp(time.time())


def run(args, config, prog_args):
    import socket
    import subprocess
    import daemon

    UNCH = {pygit2.GIT_STATUS_CURRENT, pygit2.GIT_STATUS_IGNORED}

    name = args.name
    repo = pygit2.Repository('.')

    with shelve.open('.em', writeback=True) as emdb:
        if name in emdb:
            newp = input(f'Experiment {name} already exists. Recreate? [yN] ')
            if newp.lower() != 'y':
                exit()
            _cleanup(name, emdb, repo)
            br = None

        try:
            br = repo.lookup_branch(name)
        except pygit2.GitError:
            br = None

    tracked_exts = set(config['experiment']['track_files'].split(','))
    has_src_changes = has_changes = False
    for path, status in repo.status().items():
        ext = os.path.splitext(os.path.basename(path))[1][1:]
        changed = status not in UNCH
        has_changes = has_changes or changed
        if ext in tracked_exts:
            has_src_changes = has_src_changes or changed

    saved_state = None
    base_commit = head_commit = repo.head.target  # an Oid
    sig = repo.default_signature
    if br is not None:
        if br.is_checked_out():
            return _die('error: cannot run experiment on checked out branch')
        if has_src_changes:
            return _die('error: not updating existing branch with source changes')
        base_commit = br.target
        # libgit only creates worktree from head commit, so move to the branch
        saved_state = repo.stash(sig, include_untracked=True)
        repo.reset(base_commit, pygit2.GIT_RESET_HARD)
        br.delete()
        br = None

    exper_dir = _expath(name)
    repo.add_worktree(name, exper_dir)

    if has_src_changes:
        repo.index.add_all([f'*.{ext}' for ext in tracked_exts])
        snap_tree = repo.index.write_tree()  # an Oid
        repo.create_commit(f'refs/heads/{name}', sig, sig,
                           f'setup experiment: {name}', snap_tree,
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
                                   stdin=sys.stdin, stdout=sys.stdout,
                                   stderr=sys.stderr)
            with shelve.open('.em', writeback=True) as emdb:
                emdb[name] = {
                    'started': _tstamp(),
                    'status': 'running',
                    'pid': job.pid,
                    'hostname': socket.getfqdn(),
                }
                if args.gpu:
                    emdb[name]['gpu'] = args.gpu
            job.wait()
            with shelve.open('.em', writeback=True) as emdb:
                status = 'completed' if job.returncode == 0 else 'error'
                emdb[name]['status'] = status
        except KeyboardInterrupt:
            with shelve.open('.em', writeback=True) as emdb:
                emdb[name]['status'] = 'interrupted'
        finally:
            with shelve.open('.em', writeback=True) as emdb:
                del emdb[name]['pid']
                if 'gpu' in emdb[name]:
                    del emdb[name]['gpu']
                emdb[name]['ended'] = _tstamp()

    if args.bg:
        curdir = os.path.abspath(os.curdir)
        with daemon.DaemonContext(working_directory=curdir):
            _run_job()
    else:
        _run_job()


def clean(args, config, extra_args):
    repo = pygit2.Repository('.')

    with shelve.open('.em', writeback=True) as emdb:
        _cleanup(args.name, emdb, repo)


def ls(args, config, extra_args):
    if args.filter:
        fk, fv = args.filter.split('=')

        def filt(stats):
            return fk in stats and stats[fk] == fv

    cols = shutil.get_terminal_size((80, 20)).columns
    with shelve.open('.em') as emdb:
        if args.filter:
            names = [n for n, s in sorted(emdb.items()) if filt(s)]
        else:
            names = sorted(emdb.keys())
        if not names:
            return

    linewidth = -1
    line_names = []
    for name in names:
        if linewidth + len(name) >= cols:
            linewidth = -2
            sys.stdout.write('\n')
        sys.stdout.write(f'{name}  ')
        linewidth += len(name) + 2
    sys.stdout.write('\n')
    sys.stdout.flush()


def show(args, config, extra_args):
    import pickle
    import pprint

    name = args.name

    with shelve.open('.em') as emdb:
        if name not in emdb:
            return _die(f'error: no experiment named "{name}"')
        for k, v in sorted(emdb[name].items()):
            if isinstance(v, datetime.date):
                v = v.ctime()
            print(f'{k}: {v}')

    if not args.opts:
        return

    opts_path = _expath(name, 'run', 'opts.pkl')
    with open(opts_path, 'rb') as f_opts:
        print('\noptions:')
        opts = pickle.load(f_opts)
        cols = shutil.get_terminal_size((80, 20)).columns
        pprint.pprint(vars(opts), indent=2, compact=True, width=cols)


def _ps(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def ctl(args, config, extra_args):
    import signal

    name = args.name

    with shelve.open('.em') as emdb:
        if name not in emdb:
            return _die(f'error: no experiment named "{name}"')
        pid = emdb[name].get('pid')
        if not pid or not _ps(pid):
            return _die(f'error: experiment "{name}" is not running')

    cmd = args.cmd[0]
    if cmd == 'stop':
        os.kill(pid, signal.SIGINT)
    else:
        ctl_file = _expath(name, 'run', 'ctl')
        with open(ctl_file) as f_ctl:
            print(' '.join(args.cmd), file=f_ctl)


def _get_br(repo, branch_name):
    br = None
    try:
        br = repo.lookup_branch(branch_name)
    except pygit2.GitError:
        pass
    return br


def rename(args, config, extra_args):
    repo = pygit2.init_repository('.')

    name = args.name
    new_name = args.newname

    with shelve.open('.em') as emdb:
        if name not in emdb:
            return _die(f'error: no experiment named "{name}"')
        if emdb[name]['status'] == 'running':
            return _die(f'error: cannot rename running experiment')
        if new_name in emdb:
            return _die(f'error: experiment named "{new_name}" already exists')

    br = _get_br(repo, name)
    if br is None:
        return _die(f'error: no branch for experiment "{name}"?')
    if _get_br(repo, new_name) is not None:
        return _die(f'error: branch "{new_name}" already exists')

    exper_dir = _expath(name)
    new_exper_dir = _expath(new_name)
    try:
        os.rename(exper_dir, new_exper_dir)
    except:
        return _die('error: could not move experiment directory')

    try:
        br.rename(new_name)
    except pygit2.GitError:
        os.rename(new_exper_dir, exper_dir)
        return _die(f'error: could not rename branch')

    with shelve.open('.em') as emdb:
        emdb[new_name] = emdb[name]
        del emdb[name]

# ==============================================================================

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

parser_list = subparsers.add_parser('list', aliases=['ls'], help='list experiments')
parser_list.add_argument('--filter', '-f', help='filter experiments by <state>=<val>')
parser_list.set_defaults(_cmd=_ensure_proj(ls))

parser_show = subparsers.add_parser('show', help='show details about an experiment')
parser_show.add_argument('name', help='the name of the experiment')
parser_show.add_argument('--opts', action='store_true',
                         help='also print runtime options')
parser_show.set_defaults(_cmd=_ensure_proj(show))

parser_ctl = subparsers.add_parser('ctl', help='control a running experiment')
parser_ctl.add_argument('name', help='the name of the experiment')
parser_ctl.add_argument('cmd', nargs='+', help='the control signal to send')
parser_ctl.set_defaults(_cmd=_ensure_proj(ctl))

parser_ctl = subparsers.add_parser('rename', aliases=['mv'],
                                   help='rename an experiment')
parser_ctl.add_argument('name', help='the name of the experiment')
parser_ctl.add_argument('newname', help='the new name of the experiment')
parser_ctl.set_defaults(_cmd=_ensure_proj(rename))

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
    ret = args._cmd(args, config, extra_args)
except KeyboardInterrupt:
    ret = 0

exit(ret)
