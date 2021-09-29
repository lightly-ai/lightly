set -e

# Use this script to install lightly at the current branch
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
echo $BRANCH_NAME
pip uninstall -y lightly
pip install "git+https://github.com/lightly-ai/lightly.git@$BRANCH_NAME"