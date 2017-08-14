"""Experiment Manager: A tool for managing deep learning experiments."""
import argparse
import datetime
import os
import shelve
import shutil
import sys

import pygit2


GIT_UNCH = {pygit2.GIT_STATUS_CURRENT, pygit2.GIT_STATUS_IGNORED}

E_BRANCH_EXISTS = 'error: branch "{}" already exists'
E_CHECKED_OUT = 'error: cannot run experiment on checked out branch'
E_CANT_CLEAN = 'error: could not clean up {}'
E_IS_NOT_RUNNING = 'error: experiment "{}" is not running'
E_IS_RUNNING = 'error: experiment "{}" is already running'
E_MODIFIED_SRC = 'error: not updating existing branch with source changes'
E_MOVE_DIR = 'error: could not move experiment directory'
E_NAME_EXISTS = 'error: experiment named "{}" already exists'
E_NO_BRANCH = 'error: no branch for experiment "{}"?'
E_NO_EXIST = 'error: experiment "{}" no longer exists'
E_NO_EXP = 'error: no experiment named "{}"'
E_OTHER_MACHINE = 'error: experiment "{}" is not running on this machine'
E_RENAME_BRANCH = 'error: could not rename branch'
E_RENAME_RUNNING = 'error: cannot rename running experiment'

RUN_RECREATE_PROMPT = 'Experiment {} already exists. Recreate? [yN] '

LI = '* {}'
CLEAN_NEEDS_FORCE = 'The following experiments require --force to be removed:'
CLEAN_PREAMBLE = 'The following experiments will be removed:'
CLEAN_SNAP_PREAMBLE = 'The following experiments\' snaps will be removed:'
CLEAN_PROMPT = 'Clean up {:d} experiments? [yN] '
CLEAN_SNAPS_PROMPT = 'Clean up snaps of {:d} experiments? [yN] '
LI_RUNNING = LI + ' (running)'


def _die(msg, status=1):
    print(msg, file=sys.stderr)
    return status


def _ensure_proj(cb):
    def _docmd(*args, **kwargs):
        if not os.path.isfile('.em.db'):
            curdir = os.path.abspath('.')
            return _die(f'error: "{curdir}" is not a project directory')
        cb(*args, **kwargs)
    return _docmd


def _expath(*args):
    return os.path.abspath(os.path.join('experiments', *args))


def proj_create(args, config, _extra_args):
    """Creates a new em-managed project."""
    tmpl_repo = config['project']['template_repo']
    try:
        pygit2.clone_repository(tmpl_repo, args.dest)

        # delete history of template
        shutil.rmtree(os.path.join(args.dest, '.git'), ignore_errors=True)
        pygit2.init_repository(args.dest)
    except ValueError:
        pass  # already in a repo

    for em_dir in ['experiments', 'data']:
        dpath = os.path.join(args.dest, em_dir)
        if not os.path.isdir(dpath):
            os.mkdir(dpath)

    shelve.open(os.path.join(args.dest, '.em')).close()


def _cleanup(name, emdb, repo):
    exper_dir = _expath(name)
    if name not in emdb and not os.path.isdir(exper_dir):
        return
    if os.path.isdir(exper_dir):
        shutil.rmtree(exper_dir)
    try:
        worktree = repo.lookup_worktree(name)
        if worktree is not None:
            worktree.prune(True)
    except pygit2.GitError:
        pass
        # worktree_dir = os.path.join('.git', 'worktrees', name)
        # if os.path.isdir(worktree_dir):
        #     shutil.rmtree(exper_dir)
    try:
        br = repo.lookup_branch(name)
        if br is not None:
            br.delete()
    except pygit2.GitError:
        pass
    if name in emdb:
        del emdb[name]


def _cleanup_snaps(name, _emdb, _repo):
    exper_dir = _expath(name)
    snaps_dir = os.path.join(exper_dir, 'run', 'snaps')
    if os.path.isdir(snaps_dir):
        shutil.rmtree(snaps_dir)
        os.mkdir(snaps_dir)


def _tstamp():
    import time
    return datetime.datetime.fromtimestamp(time.time())


def _run_job(name, config, gpu=None, prog_args=None, background=False):
    import socket
    import subprocess
    import daemon

    exper_dir = _expath(name)

    runem_cmd = ([config['experiment']['prog']] +
                 config['experiment']['prog_args'] +
                 (prog_args or []))
    env = os.environ
    if gpu:
        env['CUDA_VISIBLE_DEVICES'] = gpu

    def _do_run_job():
        try:
            job = subprocess.Popen(runem_cmd, cwd=exper_dir, env=env,
                                   stdin=sys.stdin, stdout=sys.stdout,
                                   stderr=sys.stderr)
            with shelve.open('.em', writeback=True) as emdb:
                emdb[name] = {
                    'started': _tstamp(),
                    'status': 'running',
                    'pid': job.pid,
                    'hostname': socket.getfqdn(),
                }
                if gpu:
                    emdb[name]['gpu'] = gpu
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
                emdb[name]['ended'] = _tstamp()

    if background:
        curdir = os.path.abspath(os.curdir)
        with daemon.DaemonContext(working_directory=curdir):
            _do_run_job()
    else:
        _do_run_job()


def run(args, config, prog_args):
    """Run an experiment."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    name = args.name
    repo = pygit2.Repository('.')

    with shelve.open('.em', writeback=True) as emdb:
        exp_info = emdb.get(name)
        if exp_info:
            if exp_info['status'] == 'running':
                return _die(E_IS_RUNNING.format(name))
            newp = input(RUN_RECREATE_PROMPT.format(name))
            if newp.lower() != 'y':
                return
            _cleanup(name, emdb, repo)
            br = None
        emdb[name] = {}

        try:
            br = repo.lookup_branch(name)
        except pygit2.GitError:
            br = None

    tracked_exts = set(config['experiment']['track_files'].split(','))
    has_src_changes = has_changes = False
    for filepath, status in repo.status().items():
        ext = os.path.splitext(os.path.basename(filepath))[1][1:]
        changed = status not in GIT_UNCH
        has_changes = has_changes or changed
        if ext in tracked_exts:
            has_src_changes = has_src_changes or changed

    if has_src_changes:
        repo.index.add_all([f'*.{ext}' for ext in tracked_exts])
        snap_tree = repo.index.write_tree()  # an Oid

    base_commit = head_commit = repo.head.target   # an Oid
    sig = repo.default_signature

    if has_src_changes:
        with shelve.open('.em') as emdb:
            for existing_name in emdb:
                existing_br = repo.lookup_branch(existing_name)
                if existing_br is None:
                    continue
                existing_ci = existing_br.get_object()
                if existing_ci and existing_ci.tree_id == snap_tree:
                    base_commit = existing_ci.id
                    break

    if br is not None:  # turn existing branch into an experiment
        if br.is_checked_out():
            return _die(E_CHECKED_OUT)
        if has_src_changes:
            return _die(E_MODIFIED_SRC)
        base_commit = br.target
        br.delete()
        br = None

    saved_state = None
    if has_src_changes:
        saved_state = repo.stash(sig, include_untracked=True)
    repo.reset(base_commit, pygit2.GIT_RESET_HARD)

    exper_dir = _expath(name)
    repo.add_worktree(name, exper_dir)

    if has_src_changes and base_commit == head_commit:
        # create a snapshot and move the worktree branch to it
        msg = args.desc or 'setup experiment'
        repo.create_commit(f'refs/heads/{name}', sig, sig, msg,
                           snap_tree, [base_commit])
        # update the workdir to match updated index
        workdir = pygit2.Repository(exper_dir)
        workdir.reset(workdir.head.target, pygit2.GIT_RESET_HARD)

    os.symlink(os.path.abspath('data'), os.path.join(exper_dir, 'data'),
               target_is_directory=True)

    repo.reset(head_commit, pygit2.GIT_RESET_HARD)
    if saved_state:
        repo.stash_pop()

    return _run_job(name, config, args.gpu, prog_args, args.background)


def resume(args, config, prog_args):
    """Resume a stopped experiment."""
    name = args.name

    repo = pygit2.Repository('.')

    with shelve.open('.em') as emdb:
        if name not in emdb:
            return _die(E_NO_EXP.format(name))
        info = emdb[name]
        if 'pid' in info or info.get('status') == 'running':
            return _die(E_IS_RUNNING.format(name))
        try:
            repo.lookup_branch(name)
        except pygit2.GitError:
            return _die(E_NO_EXIST.format(name))

    prog_args.extend(['--resume', args.epoch])

    return _run_job(name, config, args.gpu, prog_args, args.background)


def _print_sorted(lines, tmpl=LI):
    print('\n'.join(map(tmpl.format, sorted(lines))))


def clean(args, _config, _extra_args):
    """Clean up experiments."""
    from fnmatch import fnmatch
    repo = pygit2.Repository('.')

    cleanup = _cleanup_snaps if args.snaps else _cleanup

    with shelve.open('.em', writeback=True) as emdb:
        matched = set()
        needs_force = set()
        for name in emdb:
            is_match = sum(fnmatch(name, patt) for patt in args.name)
            is_excluded = sum(fnmatch(name, patt) for patt in args.exclude)
            if not is_match or is_excluded:
                continue
            matched.add(name)
            info = emdb[name]
            if 'pid' in info or info.get('status') == 'running':
                needs_force.add(name)
        if not matched:
            return
        clean_noforce = matched - needs_force
        to_clean = clean_noforce if not args.force else matched
        if len(args.name) == 1 and args.name[0] in to_clean:
            cleanup(args.name[0], emdb, repo)
            return

        if to_clean:
            print(CLEAN_SNAP_PREAMBLE if args.snaps else CLEAN_PREAMBLE)
            _print_sorted(clean_noforce)
            if args.force:
                _print_sorted(needs_force, tmpl=LI_RUNNING)
        if needs_force and not args.force:
            print(CLEAN_NEEDS_FORCE)
            _print_sorted(needs_force)

        if not to_clean:
            return

        prompt = CLEAN_SNAPS_PROMPT if args.snaps else CLEAN_PROMPT
        cleanp = input(prompt.format(len(to_clean)))
        if cleanp.lower() != 'y':
            return
        for name in to_clean:
            try:
                cleanup(name, emdb, repo)
            except OSError:
                print(E_CANT_CLEAN.format(name))


def list_experiments(args, _config, _extra_args):
    """List experiments."""
    if args.filter:
        filter_key, filter_value = args.filter.split('=')

        def _filt(stats):
            return filter_key in stats and stats[filter_key] == filter_value

    cols = shutil.get_terminal_size((80, 20)).columns
    with shelve.open('.em') as emdb:
        if args.filter:
            names = [name
                     for name, info in sorted(emdb.items()) if _filt(info)]
        else:
            names = sorted(emdb.keys())
        if not names:
            return

    linewidth = -1
    for name in names:
        if linewidth + len(name) + 2 >= cols:
            linewidth = -2
            sys.stdout.write('\n')
        sys.stdout.write(f'{name}    ')
        linewidth += len(name) + 2
    sys.stdout.write('\n')
    sys.stdout.flush()


def show(args, _config, _extra_args):
    """Show details about an experiment."""
    import pickle
    import pprint

    name = args.name

    with shelve.open('.em') as emdb:
        if name not in emdb:
            return _die(E_NO_EXP.format(name))
        for info_name, info_val in sorted(emdb[name].items()):
            if isinstance(info_val, datetime.date):
                info_val = info_val.ctime()
            print(f'{info_name}: {info_val}')

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


def ctl(args, _config, _extra_args):
    """Send a command to a running experiment."""
    import signal

    name = args.name

    with shelve.open('.em') as emdb:
        if name not in emdb:
            return _die(E_NO_EXP.format(name))
        pid = emdb[name].get('pid')
        if not pid:
            return _die(E_IS_NOT_RUNNING.format(name))
        if not _ps(pid):
            return _die(E_OTHER_MACHINE.format(name))

    cmd = args.cmd[0]
    if cmd == 'stop':
        os.kill(pid, signal.SIGINT)
    else:
        ctl_file = _expath(name, 'run', 'ctl')
        with open(ctl_file, 'w') as f_ctl:
            print(' '.join(args.cmd), file=f_ctl)


def _get_br(repo, branch_name):
    br = None
    try:
        br = repo.lookup_branch(branch_name)
    except pygit2.GitError:
        pass
    return br


def rename(args, _config, _extra_args):
    """Rename an experiment."""
    # pylint: disable=too-many-return-statements
    repo = pygit2.init_repository('.')

    name = args.name
    new_name = args.newname

    with shelve.open('.em') as emdb:
        if name not in emdb:
            return _die(E_NO_EXP.format(name))
        if emdb[name]['status'] == 'running':
            return _die(E_RENAME_RUNNING)
        if new_name in emdb:
            return _die(E_NAME_EXISTS.format(new_name))

    br = _get_br(repo, name)
    if br is None:
        return _die(E_NO_BRANCH.format(name))
    if _get_br(repo, new_name) is not None:
        return _die(E_BRANCH_EXISTS.format(new_name))

    exper_dir = _expath(name)
    new_exper_dir = _expath(new_name)
    try:
        os.rename(exper_dir, new_exper_dir)
    except OSError:
        return _die(E_MOVE_DIR)

    try:
        br.rename(new_name)
    except pygit2.GitError:
        os.rename(new_exper_dir, exper_dir)
        return _die(E_RENAME_BRANCH)

    with shelve.open('.em') as emdb:
        emdb[new_name] = emdb[name]
        del emdb[name]


def main():
    """Runs the program."""
    parser = argparse.ArgumentParser(
        description='Manage projects and experiments.')
    parser.add_argument('--config', '-c', help='path to config file')
    subparsers = parser.add_subparsers()

    parser_create = subparsers.add_parser('proj', help='create a new project')
    parser_create.add_argument('dest', help='the project destination')
    parser_create.set_defaults(em_cmd=proj_create)

    parser_run = subparsers.add_parser('run', help='run an experiment')
    parser_run.add_argument('name', help='the name of the experiment')
    parser_run.add_argument('--gpu', '-g',
                            help='CSV ids of gpus to use. none = all')
    parser_run.add_argument('--background', '-bg', action='store_true',
                            help='run the experiment in the background')
    parser_run.add_argument('--desc',
                            help='a short description of any source changes')
    parser_run.set_defaults(em_cmd=_ensure_proj(run))

    parser_ctl = subparsers.add_parser('ctl',
                                       help='control a running experiment')
    parser_ctl.add_argument('name', help='the name of the experiment')
    parser_ctl.add_argument('cmd', nargs='+',
                            help='the control signal to send')
    parser_ctl.set_defaults(em_cmd=_ensure_proj(ctl))

    parser_run = subparsers.add_parser('resume',
                                       help='resume existing experiment')
    parser_run.add_argument('name', help='the name of the experiment')
    parser_run.add_argument('epoch', help='the epoch from which to resume')
    parser_run.add_argument('--gpu', '-g',
                            help='CSV ids of gpus to use. none = all')
    parser_run.add_argument('--background', '-bg', action='store_true',
                            help='resume the experiment into the background')
    parser_run.set_defaults(em_cmd=_ensure_proj(resume))

    parser_list = subparsers.add_parser('list', aliases=['ls'],
                                        help='list experiments')
    parser_list.add_argument('--filter', '-f',
                             help='filter experiments by <state>=<val>')
    parser_list.set_defaults(em_cmd=_ensure_proj(list_experiments))

    parser_show = subparsers.add_parser('show',
                                        help='show details of an experiment')
    parser_show.add_argument('name', help='the name of the experiment')
    parser_show.add_argument('--opts', action='store_true',
                             help='also print runtime options')
    parser_show.set_defaults(em_cmd=_ensure_proj(show))

    parser_clean = subparsers.add_parser('clean',
                                         help='clean up an experiment')
    parser_clean.add_argument('name', nargs='+',
                              help='patterns of experiments to remove')
    parser_clean.add_argument('--exclude', '-e', nargs='+', default=[],
                              help='patterns of experiments to keep')
    parser_clean.add_argument('--force', '-f', action='store_true')
    parser_clean.add_argument('--snaps', '-s', action='store_true',
                              help='just empty snaps directory?')
    parser_clean.set_defaults(em_cmd=_ensure_proj(clean))

    parser_ctl = subparsers.add_parser('rename', aliases=['mv'],
                                       help='rename an experiment')
    parser_ctl.add_argument('name', help='the name of the experiment')
    parser_ctl.add_argument('newname', help='the new name of the experiment')
    parser_ctl.set_defaults(em_cmd=_ensure_proj(rename))

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
        ret = args.em_cmd(args, config, extra_args)
    except KeyboardInterrupt:
        ret = 0

    exit(ret)

if __name__ == '__main__':
    main()
