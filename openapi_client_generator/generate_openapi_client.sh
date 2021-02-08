PATH_TO_SPEC="../../lightly-core/openapi/spec/api.yml"
PATH_FROM_BASE="lightly/openapi_generated"
PATH_TO_OUTPUT="../$PATH_FROM_BASE"
#npm install -g swagger-codegen
swagger-codegen generate -l python -i $PATH_TO_SPEC  -o $PATH_TO_OUTPUT
cd $PATH_TO_OUTPUT
rm -R docs
rm -R test

PATTERN_TO_REPLACE="swagger_client"
PATTERN_REPLACING="${PATH_FROM_BASE/"/"/"."}/$PATTERN_TO_REPLACE"
find . -type f -name "*.py" -print0 | xargs -0 sed -i '' -e 's/PATTERN_TO_REPLACE/PATTERN_REPLACING/g'


