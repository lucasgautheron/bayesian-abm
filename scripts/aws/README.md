# Shared AWS instance

The workshop uses one shared `c7a.16xlarge` instance in `us-west-2` for
`/simulate`, `/inference`, `/model-comparison`, and `/report`. Those Cursor
commands mirror the local repository into an isolated remote folder, run with
`--cpus 16`, and copy `output/` and `reports/` back. `/simulate`,
`/inference`, and `/model-comparison` fall back to the local machine
(`--cpus 4`) if SSH is unavailable; `/report` does not.

Direct CLI scripts (`python scripts/simulate.py`, and so on) always run
locally.

The live instance ID and the shared SSH private key live in S3
(`s3://bayesian-modelling-workshop-292651677991/workshop/`). Anyone with
workshop AWS credentials picks up a replaced instance without updating the
repository.

## Commands

Run these from the repository root. Cursor remote commands need only
`requirements-remote.txt`; instructor setup and local CLI need the full
project environment.

| Task | Command |
| --- | --- |
| Create the instance and conda environment | `python scripts/aws/instance.py setup` |
| Finish conda setup on a running instance | `python scripts/aws/instance.py provision` |
| Publish the instance record and workshop SSH key to S3 | `python scripts/aws/instance.py publish` |
| Start a stopped instance | `python scripts/aws/instance.py start` |
| Stop the instance | `python scripts/aws/instance.py stop` |
| Terminate the instance | `python scripts/aws/instance.py destroy` |
| Probe SSH and the remote conda environment | `python scripts/aws/remote.py check` |
| Run a workshop command remotely | `python scripts/aws/remote.py run --fallback-local -- python scripts/simulate.py MODEL` |

`setup` accepts `--region`, `--instance-type`, and `--cpus` (the remote
default stored in S3). `stop` preserves the disk and the conda environment,
and it interrupts every user's remote jobs.

## Instructor setup

Once, from an account that can create EC2 resources:

```bash
python scripts/aws/instance.py setup
```

`setup` creates one workshop SSH key, writes it and the instance record to
S3, launches the instance with that public key, and installs the conda
environment. Students do not commit instance metadata or SSH keys.

If the instance already exists and only S3 is missing the record or key:

```bash
python scripts/aws/instance.py publish
```

```bash
python scripts/aws/instance.py destroy
```

removes the instance, its security group, and the S3 instance record. The
workshop SSH key stays in S3 for the next setup.

## Everyday start and stop

Anyone with workshop credentials can start or stop the existing instance:

```bash
python scripts/aws/instance.py start
python scripts/aws/instance.py stop
```

`/test` reports three checks in order: SSH key access, whether the
instance is running, and SSH connectivity. A stopped instance fails the
instance check and skips the connection check; the instance is not
started automatically.

## Access and isolation

Each person uses `~/.dallingerconfig`. Everyone SSHs with the same
workshop key from S3, as `ubuntu`. If `remote.py check` works on one
machine, it works for anyone with those credentials.

```ini
[AWS Access]
aws_access_key_id = ...
aws_secret_access_key = ...
```

The workshop instance stays in `us-west-2` even if that file sets
`aws_region`. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` override
the matching keys in the file.

Per-workstation folders stay under `/home/ubuntu/users/<id>`, using the
gitignored id in `.config/aws-remote-user` (for example
`lucasgautheron-a3f2`).

Override the bucket with `AWS_WORKSHOP_S3_BUCKET` if needed.

Students also need `ssh`, `rsync`, and the thin remote-helper install:

```bash
python -m pip install -r requirements-remote.txt
```

That file is only `boto3`. Do not install `requirements.txt` just to run
Cursor commands on the shared instance.

## IAM

Everyone:

- `ec2:DescribeInstances`
- `ec2:StartInstances`
- `ec2:StopInstances`
- `s3:GetObject` on `bayesian-modelling-workshop-292651677991`

Instructor `setup` / `destroy` / `publish` also needs:

- `ec2:RunInstances`
- `ec2:TerminateInstances`
- `ec2:CreateSecurityGroup`
- `ec2:AuthorizeSecurityGroupIngress`
- `ec2:DeleteSecurityGroup`
- `ec2:DescribeVpcs`
- `ec2:DescribeSubnets`
- `ec2:DescribeSecurityGroups`
- `ec2:CreateTags`
- `ec2:DescribeImages`
- `s3:CreateBucket`
- `s3:PutBucketPublicAccessBlock`
- `s3:PutObject`
- `s3:DeleteObject`
