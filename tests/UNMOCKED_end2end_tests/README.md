This repository contains scripts to test the python package with a server. 

## Testing the Server API with CLI commands and active learning
You only need an account on the server.
Once you have a token from our production server `https://app.lightly.ai`, you can run:
```bash
cd ../../../lightly # ensure you are in the top directory
pip uninstall lightly -y

pip install . 
bash tests/UNMOCKED_end2end_tests/run_all_unmocked_tests.sh TOKEN
```

## Testing the Server API with CLI commands
You only need an account on the server and a dataset.
Once you have a token from our production server `https://app.lightly.ai`, you can run:
```bash
bash test_api_on_branch.sh path/to/dataset TOKEN
```

## Testing the Server API with active learning
You only need an account on the server and a dataset.
Once you have a token from our production server `https://app.lightly.ai`, you can run:

```bash
python tests/UNMOCKED_end2end_tests/test_api.py path/to/dataset TOKEN
```

If you want to test on another server, e.g. staging, get your token from there and then run:
```bash
LIGHTLY_SERVER_LOCATION=https://api-staging.lightly.ai && \
python tests/UNMOCKED_end2end_tests/test_api.py path/to/dataset TOKEN_FROM_STAGING

```

## Testing the API latency
This needs a token, but no dataset
```bash
TOKEN="MY_TOKEN" && python tests/UNMOCKED_end2end_tests/test_api_latency.py
```

## Testing the upload speed
Use the pycharm profile with yappi to run the function [benchmark_upload.py:benchmark_upload()](benchmark_upload.py). 
You can use the following script for example: [call_benchmark_upload](call_benchmark_upload.py)