from typing import Type


def patch_rest_client(rest_client: Type):
    """

    Patches the rest client to flatten out array query parameters.

    Example:
        query_params: [('labels', ['AAA', 'BBB'])]
        new_query_params: [('labels', 'AAA'), ('labels', 'BBB')]
        url part in query: "labels=AAA&labels=BBB"

        Without this patch, the query_params would be translated into
        "labels=['AAA', 'BBB']" in the url itself, which fails.

    Args:
        rest_client:
            Must be the class swagger_client.rest.RESTClientObject to patch.
            Note: it must be the class itself, not an instance of it.

    Returns:

    """
    request = rest_client.request

    def request_patched(self, method, url, query_params=None, headers=None,
                    body=None, post_params=None, _preload_content=True,
                    _request_timeout=None):
        if query_params is not None:
            new_query_params = []
            for name, value in query_params:
                if isinstance(value, list):
                    new_query_params.extend([(name, val) for val in value])
                else:
                    new_query_params.append((name, value))
            query_params = new_query_params
        return request(self, method=method, url=url, query_params=query_params, headers=headers, body=body, post_params=post_params, _preload_content=_preload_content, _request_timeout=_request_timeout)

    rest_client.request = request_patched