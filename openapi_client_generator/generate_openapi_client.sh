cd ../
PATH_TO_SPEC="../lightly-core/openapi/spec/api.yml"

# install swagger-codegen following the instructions at https://github.com/swagger-api/swagger-codegen#compatibility
swagger-codegen generate -l python -i $PATH_TO_SPEC  -o lightly/openapi_generated  --template-dir openapi_client_generator/python --template-engine mustache

cd lightly/openapi_generated
rm -R docs
rm -R test

# replaces of imports
find . -type f -name "*.py" -print0 | xargs -0 sed -i '' -e 's/swagger_client/lightly.openapi_generated.swagger_client/g'
find . -type f -name "*.py" -print0 | xargs -0 sed -i '' -e 's/from api./from lightly.openapi_generated.swagger_client.api./g'

