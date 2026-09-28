# Rate limits

Rate limits protect the API from bursts that would degrade it for everyone.

## Default limits

Every workspace may make 600 requests per minute and run 20 requests
concurrently. Enterprise plans can raise both limits; contact your account
manager.

## When you are limited

A limited request fails with status 429 and a `Retry-After` header giving the
number of seconds to wait. Retry after that interval rather than immediately.
Clients that retry at once are limited again and extend their own backoff.

## Best practice

Use exponential backoff with jitter, keep concurrency below the workspace
limit, and batch writes where the endpoint supports it. The batch endpoints
accept up to 100 items per request and count as a single request.
