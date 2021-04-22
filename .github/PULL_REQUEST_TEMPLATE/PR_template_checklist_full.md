closes #issue_number

## Description
- [ ] My change is breaking
Please_describe_what_you_changed_and_why___You_do_not_need_to_repeat_stuff_from_the_issue

## Tests
- [ ] My change is covered by existing tests
- [ ] My change needs new tests
- [ ] I have added/adapted tests accordingly.
- [ ] I have manually tested the change. 

If applicable, describe the manual test procedure, e.g:
```bash
pip uninstall lightly
export BRANCH_NAME="branch_name"
pip install "git+https://github.com/lightly-ai/lightly.git@$BRANCH_NAME"
lightly-cli_do_something_command
```

## Documentation
- [ ] I have added docstrings to all changed/added public functions/methods.
- [ ] My change requires a change to the documentation ( `.rst` files).
- [ ] I have updated the documentation accordingly.
- [ ] The autodocs update the documentation accordingly.`

## Improvements put into another issue:
- #issue_number

## Issues covering the breaking change:
- #link_to_issue_in_other_repo to adapt the other side of the breaking change