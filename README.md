# experiment manager

`em` is a tool that facilitates deep learning experimentation using a [Git worktree](https://git-scm.com/docs/git-worktree) workflow.

The use case is when you want to test a number of small changes that are not worth implementing as into full-fledged options but must still be recorded, nonetheless.
`em` allows this by creating a separate worktree for each experiment; this work tree is essentially a snapshot of the configuration at the time the experiment was run.
Additionally, since each experiment has its own branch, interesting options can be merged back into the main branch.

For instance,
```
em proj test_proj
cd test_proj
git add -A && git commit -m "Initial commit."
echo "print('hello, world!')" > main.py
em run testing   # hello world!
em show testing  # {'created': 1490891521.6767356, 'status': 'completed'}
em clean testing
```

## Requirements and Installation

You will need
* Python >= 3.6
* libgit2 >= 0.26
* pygit2 >= 0.26
* `pip install daemon`

## Usage

`em --help`
