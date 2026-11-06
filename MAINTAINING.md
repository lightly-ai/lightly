# How to Maintain LightlySSL

This document is intended for maintainers of Lightly**SSL**. If you would like to
contribute, please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) document.

## Update PR from a Fork

Sometimes it is necessary to update a PR from a forked Lightly**SSL** repository,
for example to add some finishing touches (format, cleanup, docs), to fix some difficult
issue, or to finish an incomplete feature.

Let's assume there is a PR from `<username>:<branch-name>`. To update the PR, follow
these steps:

```
git remote add <username> https://github.com/<username>/lightly.git
git fetch <username>
git checkout -b <branch-name> <username>/<branch-name>
```

Now you can make changes and push them to the PR branch with `git push`.

If there's an update from LightlySSL's master branch, you can merge it into the PR branch

```
git fetch upstream
git merge upstream/master
```

## Releasing

Releases are automated with
[release-please](https://github.com/googleapis/release-please). The pull request
titles on `master` follow Conventional Commits (see
[CONTRIBUTING.md](./CONTRIBUTING.md)) and decide the next version.

1. Every push to `master` runs [release.yml](./.github/workflows/release.yml).
   release-please keeps one pull request `chore(release): LightlySSL X.Y.Z` open. It
   bumps `__version__` in `lightly/__init__.py` and adds a section to
   [CHANGELOG.md](./CHANGELOG.md).
2. Optional: to set a different version, merge any pull request with
   `Release-As: X.Y.Z` as the last line of the squash commit message.
3. Optional: polish the `X.Y.Z` section in `CHANGELOG.md` and in the pull request
   description. release-please discards these edits on the next merge to `master`, so
   merge the release pull request right after.
4. Merge the release pull request. This creates the `vX.Y.Z` tag and the GitHub release,
   which starts [release_pypi.yml](./.github/workflows/release_pypi.yml) to publish to
   PyPI.

| Type in the PR title | Changelog section | Opens a release | Bump |
|---|---|---|---|
| `feat` | Added | yes | minor |
| `fix` | Fixed | yes | patch |
| `perf`, `revert` | Changed | yes | patch |
| `refactor`, `docs`, `test`, `build`, `ci`, `chore` | hidden | no | none |
| any type with `!` | Breaking Changes | yes | major |

If PyPI publishing fails for infrastructure reasons, rerun
[release_pypi.yml](./.github/workflows/release_pypi.yml) with `workflow_dispatch`. It
builds the latest tag.

The Lightly Release Bot GitHub App creates the release pull request and the release.
`RELEASE_APP_ID` (variable) and `RELEASE_APP_PRIVATE_KEY` (secret) live in the `release`
environment of this repository.
