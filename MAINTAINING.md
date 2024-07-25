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
