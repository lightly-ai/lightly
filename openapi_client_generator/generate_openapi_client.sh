cd ../
PATH_TO_SPEC="../lightly-core/openapi/spec/api.yml"

#npm install -g swagger-codegen
swagger-codegen generate -l python -i $PATH_TO_SPEC  -o lightly/openapi_generated  -t openapi_client_generator/python

cd lightly/openapi_generated
rm -R docs
rm -R test
find . -type f -name "*.py" -print0 | xargs -0 sed -i '' -e 's/swagger_client/lightly.openapi_generated.swagger_client/g'


