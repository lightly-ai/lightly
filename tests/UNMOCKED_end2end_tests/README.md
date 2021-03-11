This repository contains scripts to test the python package with a server. 
You only need an account on the server and a dataset.

Once you have a token from our production server `https://app.lightly.ai`, you can run:

```bash
python tests/UNMOCKED_end2end_tests/test_api.py path/to/dataset TOKEN_FROM_PRODUCTION
```

If you want to test on another server, e.g. staging, get your token from there and then run:
```bash
export LIGHTLY_SERVER_LOCATION=https://api-staging.lightly.ai
python tests/UNMOCKED_end2end_tests/test_api.py path/to/dataset TOKEN_FROM_STAGING
```
