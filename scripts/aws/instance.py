"""Create, start, stop, and destroy the shared workshop EC2 instance."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import shlex
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws.support import (
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_REGION,
    DEFAULT_REMOTE_CPUS,
    SENTINEL_MINICONDA,
    SENTINEL_PROVISIONED,
    WORKSHOP_NAME,
    AwsError,
    InstanceConfig,
    clear_instance_config,
    describe_instance,
    ec2_client,
    ensure_instance_running,
    ensure_workshop_ssh_key,
    load_instance_config,
    load_instance_config_optional,
    load_local_instance_config,
    instance_state,
    open_ssh_session,
    require_live_instance,
    run_ssh,
    save_instance_config,
    translate_boto_error,
    upload_workshop_ssh_key,
    wait_for_remote_file,
    wait_for_ssh,
    wait_for_state,
)


def user_data_script(public_key: str) -> str:
    """Return cloud-init that installs Miniconda and the workshop SSH key."""

    key = public_key.strip()
    if not key:
        raise AwsError(
            "The workshop SSH public key is empty.",
            "Rerun `python scripts/aws/instance.py setup`.",
        )
    return f"""#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
apt-get update
apt-get install -y --no-install-recommends \\
  ca-certificates curl rsync openssh-server
install -d -m 0777 /opt/workshop /home/ubuntu/users
install -d -m 0700 /home/ubuntu/.ssh
cat > /home/ubuntu/.ssh/authorized_keys <<'EOF'
{key}
EOF
chown -R ubuntu:ubuntu /home/ubuntu/.ssh
chmod 600 /home/ubuntu/.ssh/authorized_keys
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \\
  -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda3
/opt/miniconda3/bin/conda create -y -n bayesian-modelling python=3.12 pip
install -d -m 0755 /opt/miniconda3/envs/bayesian-modelling/etc/conda/activate.d
printf '%s\\n' 'export KERAS_BACKEND=jax' \\
  > /opt/miniconda3/envs/bayesian-modelling/etc/conda/activate.d/keras.sh
chmod -R a+rX /opt/miniconda3
chmod 0777 /opt/workshop /home/ubuntu/users
touch /opt/workshop/.miniconda-ready
"""

WORKSHOP_TAGS = (
    {"Key": "Name", "Value": WORKSHOP_NAME},
    {"Key": "Project", "Value": "bayesian-modelling"},
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage the shared workshop EC2 instance.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser(
        "setup",
        help="launch the instance and install the conda environment",
    )
    setup.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    setup.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})",
    )
    setup.add_argument(
        "--cpus",
        type=int,
        default=DEFAULT_REMOTE_CPUS,
        help="remote Cursor --cpus default stored in the instance file",
    )

    for name, help_text in (
        ("destroy", "terminate the instance and delete its security group"),
        ("start", "start a stopped shared instance"),
        ("stop", "stop the shared instance (interrupts every user's jobs)"),
        (
            "provision",
            "finish conda setup on an already-running instance",
        ),
        (
            "publish",
            "upload the instance record and workshop SSH key to S3",
        ),
    ):
        subparsers.add_parser(name, help=help_text)

    return parser.parse_args(argv)


def default_vpc_id(ec2: Any) -> str:
    response = ec2.describe_vpcs(
        Filters=[{"Name": "is-default", "Values": ["true"]}]
    )
    vpcs = response.get("Vpcs") or []
    if not vpcs:
        raise AwsError(
            "This AWS account has no default VPC in the selected region.",
            "Create a default VPC or pass a region that already has one.",
        )
    return str(vpcs[0]["VpcId"])


def public_subnet_id(ec2: Any, vpc_id: str) -> str:
    response = ec2.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "default-for-az", "Values": ["true"]},
        ]
    )
    subnets = response.get("Subnets") or []
    if not subnets:
        response = ec2.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        subnets = response.get("Subnets") or []
    if not subnets:
        raise AwsError(
            f"No subnets were found in VPC {vpc_id}.",
            "Restore the default VPC subnets and rerun setup.",
        )
    return str(subnets[0]["SubnetId"])


CANONICAL_OWNER_ID = "099720109477"
UBUNTU_NOBLE_AMI_NAME = (
    "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
)


def resolve_ubuntu_ami(ec2: Any) -> str:
    """Return the newest Ubuntu 24.04 amd64 AMI via EC2, not SSM."""

    response = ec2.describe_images(
        Owners=[CANONICAL_OWNER_ID],
        Filters=[
            {"Name": "name", "Values": [UBUNTU_NOBLE_AMI_NAME]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = response.get("Images") or []
    if not images:
        raise AwsError(
            "Could not find an Ubuntu 24.04 AMI in this region.",
            "Confirm ec2:DescribeImages is allowed and that the region "
            "publishes Canonical Ubuntu 24.04 images.",
        )
    newest = max(images, key=lambda image: str(image.get("CreationDate") or ""))
    ami = newest.get("ImageId")
    if not ami:
        raise AwsError(
            "Ubuntu 24.04 AMI lookup returned no image ID.",
            "Rerun setup or pass a region that publishes Canonical images.",
        )
    return str(ami)


def ensure_security_group(ec2: Any, vpc_id: str) -> str:
    existing = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [WORKSHOP_NAME]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    ).get("SecurityGroups") or []
    if existing:
        return str(existing[0]["GroupId"])

    created = ec2.create_security_group(
        GroupName=WORKSHOP_NAME,
        Description="SSH for the Bayesian modelling workshop instance",
        VpcId=vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "security-group",
                "Tags": list(WORKSHOP_TAGS),
            }
        ],
    )
    group_id = str(created["GroupId"])
    ec2.authorize_security_group_ingress(
        GroupId=group_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [
                    {
                        "CidrIp": "0.0.0.0/0",
                        "Description": "SSH for the workshop instance",
                    }
                ],
            }
        ],
    )
    return group_id


def reject_existing_live_instance(root: Path = ROOT) -> None:
    config = load_instance_config_optional() or load_local_instance_config(root)
    if config is None:
        return
    ec2 = ec2_client(config["region"])
    description = describe_instance(ec2, config["instance_id"])
    state = instance_state(description)
    if description is not None and state in {
        "pending",
        "running",
        "stopping",
        "stopped",
        "shutting-down",
    }:
        raise AwsError(
            f"Shared instance {config['instance_id']} already exists ({state}).",
            "Run `python scripts/aws/instance.py destroy` before setup, or "
            "use start/stop to reuse the current instance.",
        )


def provision_conda_environment(
    session: Any,
    requirements: Path,
) -> None:
    wait_for_ssh(session)
    wait_for_remote_file(session, SENTINEL_MINICONDA)
    payload = requirements.read_text(encoding="utf-8")
    result = run_ssh(
        session,
        (
            "cat > /opt/workshop/requirements.txt <<'EOF'\n"
            f"{payload}"
            "\nEOF"
        ),
        capture=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "copy failed").strip()
        raise AwsError(
            f"Could not copy requirements.txt to the instance: {detail}",
            "Confirm SSH works, then rerun setup.",
        )

    command = " && ".join(
        [
            "/opt/miniconda3/bin/conda run -n bayesian-modelling "
            "python -m pip install --upgrade pip",
            "/opt/miniconda3/bin/conda run -n bayesian-modelling "
            "python -m pip install -r /opt/workshop/requirements.txt",
            "/opt/miniconda3/bin/conda run -n bayesian-modelling "
            "python -m pip install jax",
            "sudo install -d -m 0755 "
            "/opt/miniconda3/envs/bayesian-modelling/etc/conda/activate.d",
            "printf '%s\\n' 'export KERAS_BACKEND=jax' | sudo tee "
            "/opt/miniconda3/envs/bayesian-modelling/etc/conda/activate.d/"
            "keras.sh > /dev/null",
            "sudo chmod -R a+rX /opt/miniconda3",
            f"touch {shlex.quote(SENTINEL_PROVISIONED)}",
        ]
    )
    print("Installing project dependencies in the remote conda environment.")
    result = run_ssh(session, command, capture=False)
    if result.returncode != 0:
        raise AwsError(
            "Remote conda environment provisioning failed.",
            "Inspect the SSH output above and rerun setup after destroy.",
        )


def setup_instance(
    *,
    region: str,
    instance_type: str,
    cpus: int,
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> InstanceConfig:
    if cpus < 1:
        raise AwsError(
            "Remote --cpus must be a positive integer.",
            "Rerun setup with --cpus 16.",
        )
    reject_existing_live_instance(root)
    workshop_identity, workshop_public = ensure_workshop_ssh_key(root)
    upload_workshop_ssh_key(workshop_identity, workshop_public)
    announce("Published the shared workshop SSH key to S3.")
    ec2 = ec2_client(region)
    vpc_id = default_vpc_id(ec2)
    subnet_id = public_subnet_id(ec2, vpc_id)
    ami_id = resolve_ubuntu_ami(ec2)
    security_group_id = ensure_security_group(ec2, vpc_id)
    announce(
        f"Launching {instance_type} in {region} from {ami_id}."
    )
    launched = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        SubnetId=subnet_id,
        SecurityGroupIds=[security_group_id],
        UserData=user_data_script(workshop_public),
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 50,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": list(WORKSHOP_TAGS),
            },
            {
                "ResourceType": "volume",
                "Tags": list(WORKSHOP_TAGS),
            },
        ],
    )
    instance_id = str(launched["Instances"][0]["InstanceId"])
    config = InstanceConfig(
        instance_id=instance_id,
        region=region,
        instance_type=instance_type,
        security_group_id=security_group_id,
        default_cpus=cpus,
    )
    uri = save_instance_config(config, root)
    announce(f"Published {uri} with instance {instance_id}.")
    announce("Waiting for the instance to start.")
    wait_for_state(ec2, instance_id, "instance_running")
    provision_existing_instance(config, root=root, announce=announce)
    return config


def provision_existing_instance(
    config: InstanceConfig | None = None,
    *,
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> None:
    """Install project dependencies on a running instance over SSH."""

    if config is None:
        config = load_instance_config(root)
    ec2 = ec2_client(config["region"])
    description = ensure_instance_running(
        ec2,
        config,
        start_if_stopped=True,
        announce=announce,
    )
    announce("Provisioning the remote conda environment.")
    with open_ssh_session(description, root=root) as session:
        provision_conda_environment(session, root / "requirements.txt")
    announce(
        "Setup is complete. The instance record and workshop SSH key are "
        "in S3. Anyone with workshop AWS credentials uses that same key."
    )


def destroy_instance(
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> None:
    config = load_instance_config(root)
    ec2 = ec2_client(config["region"])
    instance_id = config["instance_id"]
    description = describe_instance(ec2, instance_id)
    state = instance_state(description)
    if description is not None and state not in {"terminated", "not-found"}:
        announce(f"Terminating {instance_id}.")
        ec2.terminate_instances(InstanceIds=[instance_id])
        wait_for_state(ec2, instance_id, "instance_terminated")
    group_id = config["security_group_id"]
    announce(f"Deleting security group {group_id}.")
    deadline = time.monotonic() + 180
    last_error = "security group is still in use"
    while time.monotonic() < deadline:
        try:
            ec2.delete_security_group(GroupId=group_id)
            last_error = ""
            break
        except Exception as exc:
            last_error = str(exc)
            time.sleep(5)
    if last_error:
        announce(
            f"Could not delete security group {group_id}: {last_error}. "
            "Delete it in the AWS console if it is still present."
        )
    clear_instance_config(root)
    announce("Removed the instance record from S3.")


def start_instance(
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> None:
    config = load_instance_config(root)
    ec2 = ec2_client(config["region"])
    description = ensure_instance_running(
        ec2,
        config,
        start_if_stopped=True,
        announce=announce,
    )
    announce(
        f"Instance {config['instance_id']} is running "
        f"at {description.get('PublicIpAddress')}."
    )


def stop_instance(
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> None:
    config = load_instance_config(root)
    announce(
        "Stopping the shared instance will interrupt every user's remote jobs."
    )
    ec2 = ec2_client(config["region"])
    description = require_live_instance(ec2, config)
    state = instance_state(description)
    if state == "stopped":
        announce(f"Instance {config['instance_id']} is already stopped.")
        return
    if state == "stopping":
        wait_for_state(ec2, config["instance_id"], "instance_stopped")
        announce(f"Instance {config['instance_id']} is stopped.")
        return
    ec2.stop_instances(InstanceIds=[config["instance_id"]])
    wait_for_state(ec2, config["instance_id"], "instance_stopped")
    announce(f"Instance {config['instance_id']} is stopped.")


def publish_workshop_access(
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> None:
    """Upload the current instance record and shared SSH key to S3."""

    identity, public_key = ensure_workshop_ssh_key(root)
    key_uri = upload_workshop_ssh_key(identity, public_key)
    config = load_instance_config_optional() or load_local_instance_config(root)
    if config is None:
        raise AwsError(
            "No instance record was found in S3 or a local instance file.",
            "Run `python scripts/aws/instance.py setup` to create the "
            "shared instance.",
        )
    uri = save_instance_config(config, root)
    announce(f"Published the instance record to {uri}.")
    announce(f"Published the workshop SSH key to {key_uri}.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "setup":
            setup_instance(
                region=args.region,
                instance_type=args.instance_type,
                cpus=args.cpus,
            )
        elif args.command == "destroy":
            destroy_instance()
        elif args.command == "start":
            start_instance()
        elif args.command == "stop":
            stop_instance()
        elif args.command == "provision":
            provision_existing_instance()
        elif args.command == "publish":
            publish_workshop_access()
        else:
            raise AwsError(
                f"Unknown command {args.command!r}.",
                "Use setup, destroy, start, stop, provision, or publish.",
            )
    except AwsError as exc:
        print(f"error: {exc.problem}", file=sys.stderr)
        print(f"Next step: {exc.resolution}", file=sys.stderr)
        return 1
    except Exception as exc:
        translated = translate_boto_error(exc)
        print(f"error: {translated.problem}", file=sys.stderr)
        print(f"Next step: {translated.resolution}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
