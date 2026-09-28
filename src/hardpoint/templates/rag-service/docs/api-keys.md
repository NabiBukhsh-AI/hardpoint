# API keys

API keys authenticate requests to the Northwind API. Each key belongs to one
workspace and carries the permissions of the role it was created with.

## Creating a key

Open **Settings > API keys** and choose **New key**. Give the key a name that
says where it will be used, such as `billing-service-prod`, and pick a role.
The key is shown once. Store it in your secret manager immediately; Northwind
cannot show it again.

## Rotating a key

Rotate a key by creating a new key, deploying it everywhere the old key is
used, and then revoking the old key. Both keys work during the overlap, so
rotation causes no downtime. We recommend rotating production keys every 90
days and immediately after anyone with access leaves the team.

## Revoking a key

Revoking a key takes effect within 60 seconds across every region. Requests
made with a revoked key fail with status 401 and the error code
`key_revoked`. Revocation cannot be undone; create a new key instead.

## Key limits

A workspace can hold at most 50 active keys. Keys that have not been used for
180 days are flagged in the dashboard so they can be reviewed and revoked.
